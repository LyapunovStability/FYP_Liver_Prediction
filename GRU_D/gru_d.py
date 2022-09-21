import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math


SEQ_LEN = 48

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class FeatureRegression(nn.Module):
    def __init__(self, input_size, device="cuda:0"):
        super(FeatureRegression, self).__init__()
        self.device = device
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size).to(self.device) - torch.eye(input_size, input_size).to(self.device)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False, device="cuda:0"):
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.device = device

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size).to(self.device)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class GRD_D(nn.Module):
    def __init__(self, input_dim, rnn_hid_size, impute_weight, label_weight, device="cuda:0"):
        super(GRD_D, self).__init__()

        self.device = device
        self.rnn_hid_size = rnn_hid_size
        self.input_dim = input_dim

        self.build()

    def build(self):


        self.rnn_cell = nn.LSTMCell(self.input_size, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size=self.input_size, output_size=self.rnn_hid_size, diag=False, device=self.device)
        self.temp_decay_x = TemporalDecay(input_size=self.input_size, output_size=input_size, diag=True, device=self.device)

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.input_size)
        self.feat_reg = FeatureRegression(input_size, device=self.device)

        self.weight_combine = nn.Linear(input_size * 2, self.input_size)

        self.dropout = nn.Dropout(p = 0.5)
        self.out = nn.Linear(self.rnn_hid_size, 1)



    def gen_aux_data(self, record_num, mask, x):

        B, L, N = mask.shape
        delta = np.zeros((B, L, N)).to(x.device)
        x_forward = np.zeros((B, L, N)).to(x.device)
        for i in range(B):
            for j in range(L):
                if j == record_num[i]:
                    break
                if j == 0:
                    x_forward[i, j, :] = x[i, j, :]
                    delta[i, j, :] = 1
                else:
                    x_forward[i, j, :] = mask[i, j, :] * x_forward[i, j, :] - (1 - mask[i, j, :]) * x_forward[i, j-1, :]
                    delta[i, j, :] = (stamp[i, j] - stamp[i, j - 1]) + (1 - M[i, j, :]) * delta[i, j - 1, :]

        return delta, x_forward



    def forward(self, x, mask, record_num, time_stamp):
        B, L, K = x.shape
        deltas, forwards = self.gen_aux_data(record_num, mask, x, time_stamp) # B, L, K
        h_mask = (time_stamp).bool().float() # B, L
        h_mask = h_mask.unsqueeze(-1).expand(-1, -1, self.rnn_hid_size)

        h = Variable(torch.zeros((B, self.rnn_hid_size))).to(x.device)

        h_state = []

        for t in range(L):
            x_t = x[:, t, :]
            m = mask[:, t, :]
            d = deltas[:, t, :]
            f = forwards[:, t, :] #上一time step observed data
            h_m = h_mask[:, t, :]
            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h_state.append(h)
            h = h * gamma_h
            x_h = m * x_t + (1 - m) * (1 - gamma_x) * f
            inputs = torch.cat([x_h, m], dim=1)
            h = self.rnn_cell(inputs, h)
            h = h * h_m + (1 - h_m) * h_state[-1]

        y_h = self.out(self.dropout(h))
        y_h = torch.sigmoid(y_h)

        return y_h



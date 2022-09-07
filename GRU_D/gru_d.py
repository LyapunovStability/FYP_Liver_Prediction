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
    def __init__(self, input_size, output_size, diag = False, device="cuda:0"):
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
    def __init__(self, rnn_hid_size, impute_weight, label_weight, device="cuda:0"):
        super(GRD_D, self).__init__()

        self.device = device
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight

        self.build()

    def build(self):
        input_size = 16 # param dim

        self.rnn_cell = nn.LSTMCell(input_size * 2 + 6, self.rnn_hid_size) # param dim + static data

        self.temp_decay_h = TemporalDecay(input_size=input_size, output_size=self.rnn_hid_size, diag=False, device=self.device)
        self.temp_decay_x = TemporalDecay(input_size=input_size, output_size=input_size, diag=True, device=self.device)

        self.hist_reg = nn.Linear(self.rnn_hid_size, input_size)
        self.feat_reg = FeatureRegression(input_size, device=self.device)

        self.weight_combine = nn.Linear(input_size * 2, input_size)

        self.dropout = nn.Dropout(p = 0.5)
        self.out = nn.Linear(self.rnn_hid_size, 1)



    def forward_HA(self, data, direct):
        # Original sequence with 24 time steps

        values = data["x"] # B, L, K
        masks = data["m"] # B, L, K
        deltas = data["delta"] # B, L, K
        forwards = data["x_forward"] # B, L, K
        h_mask = data["h_mask"]
        labels = data["y"] # 1, 2,
        static_data = data["static_data"]  # B, L, K

        is_train = 1

        labels = labels.reshape(-1, 1)
        # is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.to(self.device), c.cuda().to(self.device)

        y_loss = 0.0

        imputations = []
        h_state = []
        B, L, K = values.shape
        for t in range(L):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            f = forwards[:, t, :] #上一time step observed data
            h_m = h_mask[:, t, :]
            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h_state.append(h)

            h = h * gamma_h

            x_h = m * x + (1 - m) * (1 - gamma_x) * f

            x_h = torch.cat([x_h, static_data], dim=-1)

            inputs = torch.cat([x_h, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            h = h * h_m + (1 - h_m) * h_state[-1]




            imputations.append(x_h.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(self.dropout(h))
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce=False)
        y_loss = torch.sum(y_loss * is_train) / (is_train + 1e-5)

        y_h = F.sigmoid(y_h)

        return {'loss': y_loss, 'predictions': y_h, 'labels': labels, 'is_train': is_train}




    def run_on_batch_HA(self, data, optimizer, epoch=None):

        y_loss = 0
        ret = self.forward_HA(data, direct='forward')
        loss = ret["loss"]

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return ret

    def evaluate_on_batch_HA(self, data, optimizer, epoch=None):

        return None

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from models.param import *
import math
from math import sqrt


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
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
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
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

class RITS(nn.Module):
    def __init__(self, rnn_hid_size, impute_weight, label_weight):
        super(RITS, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight

        self.build()

    def build(self):
        input_size = feature_dim
        self.rnn_cell = nn.GRUCell(input_size * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size=input_size, output_size=self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size=input_size, output_size=input_size, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, input_size)
        self.feat_reg = FeatureRegression(input_size)

        self.weight_combine = nn.Linear(input_size * 2, input_size)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)

    def gen_aux_data(self, record_num, mask, time_stamp):

        B, L, N = mask.shape
        delta = np.zeros((B, L, N)).to(x.device)

        for i in range(B):
            for j in range(L):
                if j == record_num[i]:
                    break
                if j == 0:
                    delta[i, j, :] = 1
                else:
                    delta[i, j, :] = torch.abs(stamp[i, j] - stamp[i, j - 1]) + (1 - M[i, j, :]) * delta[i, j - 1, :]

        return delta


    def forward(self, x, mask, record_num, time_stamp, direct):

        B, L, K = x.shape

        values = x
        masks = mask
        deltas = self.gen_aux_data(record_num, mask, time_stamp)

        evals = (time_stamp).bool().float() # B, L
        h_mask = evals.unsqueeze(-1).expand(-1, -1, self.rnn_hid_size)
        eval_masks = evals.unsqueeze(-1).expand(-1, -1, K)
        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size))).to(x.device)


        x_loss = 0.0

        h_state = []


        for t in range(L):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            h_m = h_mask[:, t, :]
            e_m = eval_masks[:, t, :]
            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h_state.append(h)
            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c = m * x + (1 - m) * x_h
            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m * e_m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m * e_m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, h)
            h = h * h_m + (1 - h_m) * h_state[-1]

            imputations.append((c_c * e_m).unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)

        y_h = torch.sigmoid(y_h)

        return {"imputations": imputations, "x_loss": x_loss, "predictions": y_h}


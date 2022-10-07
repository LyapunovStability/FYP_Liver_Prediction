import torch
import torch.nn as nn

from torch.autograd import Variable

from BRITS.rits import rits


class brits(nn.Module):
    def __init__(self, input_size, rnn_hid_size, impute_weight=0.3, label_weight=1):
        super(brits, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.input_size = input_size
        self.build()

    def build(self):
        self.rits_f = rits(self.input_size, self.rnn_hid_size, self.impute_weight, self.label_weight)
        self.rits_b = rits(self.input_size, self.rnn_hid_size, self.impute_weight, self.label_weight)

    def forward(self,  x, mask, record_num, time_stamp):
        ret_f = self.rits_f(x, mask, record_num, time_stamp, 'forward')
        x, mask, record_num, time_stamp = self.reverse_input(x, mask, record_num, time_stamp)
        ret_b = self.rits_b(x, mask, record_num, time_stamp, 'backward')
        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):

        x_loss_f = ret_f['x_loss']
        x_loss_b = ret_b['x_loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = x_loss_f + x_loss_b + loss_c

        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['x_loss'] = loss
        ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f

    def reverse_input(self, x, mask, record_num, time_stamp):
        B, L, K = x.shape
        x_r = torch.zeros_like(x)
        mask_r = torch.zeros_like(mask)
        time_stamp_r = torch.zeros_like(time_stamp)
        for i in range(B):
            n = record_num[i]
            x_r[i, :n, :] = torch.flip(x[i, :n, :], dims=[0])
            mask_r[i, :n, :]  = torch.flip(mask[i, :n, :], dims=[0])
            time_stamp_r[i, :n]  = torch.flip(time_stamp[i, :n], dims=[0])

        return x_r, mask_r, record_num, time_stamp_r

    def reverse_output(self, ret, record_num):
        B, L, K = ret["imputations"].shape
        imputations = torch.zeros_like(ret["imputations"])
        for i in range(B):
            n = record_num[i]
            imputations[i, :n, :] = x[i, n::, :]

        ret["imputations"] = imputations

        return ret


    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss


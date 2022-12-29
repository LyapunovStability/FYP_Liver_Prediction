import math

import torch
import numpy as np
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / d_model)))


        self.register_buffer("pe", pe)

    def forward(self, x, time_stamp) -> torch.Tensor:
        B, L = time_stamp.shape
        pe = []
        for i in range(B):
            pe.append(self.pe[time_stamp[i].long()])
        pe = torch.stack(pe) # B, L, D
        x = math.sqrt(self.d_model) * x
        x = x + pe.requires_grad_(False)
        return x




class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, dropout_rate=0.1) -> None:
        super(EncoderBlock, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, dropout=dropout_rate, activation="gelu")

    def forward(self, x, src_key_padding_mask):
        x = x.transpose(0, 1) # L,D,B
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)
        return x


class DenseInterpolation(nn.Module):
    def __init__(self, seq_len=48, factor=4):
        """
        :param seq_len: sequence length
        :param factor: factor M
        """
        super(DenseInterpolation, self).__init__()

        W = np.zeros((factor, seq_len), dtype=np.float32)

        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1+m)) / factor), dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = torch.tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x, record_num) -> torch.Tensor:
        B, L, D = x.shape
        emb = []
        for i in range(B):
            # x_emb = torch.cat([x[i, record_num[i]-1, :], x[i, 0, :], x[i, record_num[i]//2, :]], dim=-1)
            #mean = torch.mean(x[i, :record_num[i], :], dim=0)
      
            mid = record_num[i].item()//2
            mean = torch.mean(x[i, mid:record_num[i], :], dim=0)
            # emb.append(x[i, record_num[i]-1, :])
            # emb.append(x_emb)
            emb.append(mean)
        emb = torch.stack(emb, dim=0) # B, D
        return emb


class ClassificationModule(nn.Module):
    def __init__(self, d_model, num_class=1) -> None:
        super(ClassificationModule, self).__init__()
        self.d_model = d_model
        self.fc = nn.Linear(d_model, num_class)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        pred = torch.sigmoid(x)
        return pred




import torch
import torch.nn as nn
from SAND.modules import *

"""
Simply Attend and Diagnose model

The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)

`Attend and Diagnose: Clinical Time Series Analysis Using Attention Models <https://arxiv.org/abs/1711.03905>`_
Huan Song, Deepta Rajan, Jayaraman J. Thiagarajan, Andreas Spanias
"""


class EncoderLayerForSAnD(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2) -> None:
        super(EncoderLayerForSAnD, self).__init__()
        self.d_model = d_model

        self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        self.positional_encoding = modules.PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x, time_stamp, mask=None):
        x = x.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)

        x = self.positional_encoding(time_stamp)

        for l in self.blocks:
            x = l(x, src_key_padding_mask=mask)

        return x


class SAnD(nn.Module):

    def __init__(
            self, input_features: int, seq_len: int, n_heads: int, factor: int,
            n_class: int, n_layers: int, d_model: int = 128, dropout_rate: float = 0.2
    ) -> None:
        super(SAnD, self).__init__()
        self.encoder = EncoderLayerForSAnD(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.dense_interpolation = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, factor, n_class)

    def forward(self, x, record_num, time_stamp):
        B, L, K = x.shape
        src_key_padding_mask = torch.ones(B, L).to(x.device) # N,L
        for i in range(B):
            src_key_padding_mask[i, :record_num[i]] = 0
        x = self.encoder(x, record_num, time_stamp, src_key_padding_mask)
        x = self.dense_interpolation(x, record_num)
        x = self.clf(x)
        return x

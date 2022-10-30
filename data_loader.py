import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import random


class MySet(Dataset):
    def __init__(self, path="patient_data.csv", window=1.0):
        super(MySet, self).__init__()
        path = path
        data_id = pd.read_csv(path, header=0, usecols=[0]).values
        data = pd.read_csv(path, header=0).values
        data_id = set(data_id.squeeze(-1).tolist())
        patient_data = []
        patient_time = []
        patient_label = []
        record_num = []
        max_len = 0
        for id in data_id:
            value = data[data[:, 0] == id]
            end_t = data[:, -2].astype("M8[M]").astype("int32")
            x = value[:, 2:-3]
            t = value[:, 1].astype("M8[M]").astype("int32")
            x, t = self.filter_data_window(x, t, end_t, window=window)
            t = t - t[0] + 1
            y = value[0, -1]
            record_num.append(len(t))
            max_len = max(len(t), max_len)
            patient_data.append(x)
            patient_time.append(t)
            patient_label.append(y)

        self.x = torch.zeros((len(data_id), max_len, 17))
        self.mask = torch.ones((len(data_id), max_len, 17))
        self.time_stamp = torch.zeros((len(data_id), max_len))
        self.y = torch.tensor(patient_label)
        for i in range(len(record_num)):
            num = record_num[i]
            self.x[i, :num, :] = torch.from_numpy(patient_data[i][:, :].astype("float"))
            self.mask[i, :, 2:] = 1 - (self.x[i, :, 2:] == 0.0).float()
            self.time_stamp[i, :num] = torch.from_numpy(patient_time[i][:].astype("float"))
        self.record_num = torch.tensor(record_num)


        # self.x, self.y, self.mask, self.record_num, self.time_stamp = simulate_data()

    def filter_data_window(self, x, t , end_t, window=1):
        num = 0
        for i in len(x):
            d = end_t[i] - t[i]
            if d < window * 12 :
                break

        x = x[:num, :]
        t = t[:num]

        return x, t




    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i]
        mask = self.mask[i]
        time_stamp = self.time_stamp[i]
        record_num = self.record_num[i]

        return x, y, mask, time_stamp, record_num




# def simulate_data():
#     sample_num = 200
#     max_seq = 40
#     input_size = 17
#
#     record_num = []
#     x = torch.zeros((sample_num, 40, input_size))
#     mask = torch.zeros((sample_num, 40, input_size))
#     y = []
#     time_stamp = torch.zeros((sample_num, 40))
#     for i in range(sample_num):
#         num = torch.randint(1, max_seq + 1, (1,))[0]
#         stamp_i = torch.arange(num) + 1
#         time_stamp[i, :num] = stamp_i[:]
#         y_i = torch.randint(0, 2, (1,))[0]
#         record_num.append(num)
#         y.append(y_i)
#         m = torch.randint(0, 2, (num, input_size))
#         for l in range(num):
#             if m[l, :].sum() == 0:
#                 m[l, -1] = 1
#         x_i = torch.rand((num, input_size))
#         x[i, :num, :] = x_i[:, :]
#         mask[i, :num, :] = m[:, :]
#
#     y = torch.tensor(y)
#     record_num = torch.tensor(record_num)
#
#     return x, y, mask, record_num, time_stamp



def get_dataloader(batch_size, path="patient_data.csv", time_window=0.5):
    dataset = MySet(path=path, time_window=time_window)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )


    return data_loader

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
# lab 2~18
feature_mean = [111.66981005411401, 38.877754948643066, 45.07860606054799, 45.992427244092234,
              147.12718977980745, 6.989963767105929, 115.07515184877008, 7.133366388932162,
              1.221436639591917, 1.418330594123345, 2.705382125814853, 239.36558562440754,
              13.355878455402308, 4.790074454699277, 1.985037625807247, 8.541386564955026]

feature_std = [7106.654231384638, 7.0098232213744955, 73.55326516442304, 257.11350656940874,
             201.52270095277365, 2.4558367154592307, 195.09957425653232, 1.4432099391918614,
             0.3244507933595641, 0.760913384811514, 0.9445827702888436, 97.99000430097774,
             18.951603980680332, 1.1394681379306435, 1.6808890993793133, 4.682248175889633]
# age

age_mean = 56.25664615109769
age_std = 13.459039494161564

#
feature_name = ['age', 'gender',
                'AFP', 'Albumin', 'ALT', 'AST',
                'Creatinine', 'FBS', 'GGT', 'HbA1c',
                'HDL', 'INR', 'LDL', 'PLT',
                'Total Bilirubin', 'Total Cholesterol',  'Triglyceride', 'WBC']



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
        print("Read Data-----------")
        for id in data_id:
            value = data[data[:, 0] == id]
            end_t = value[:, -2].astype("M8[M]").astype("int32")
            x = value[:, 2:-2]
            t = value[:, 1].astype("M8[M]").astype("int32")
            # print("id: ", id)
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
        print("Patient Num: ", len(self.record_num))
        print("Finish Data-----------")
        # self.x, self.y, self.mask, self.record_num, self.time_stamp = simulate_data()

    def filter_data_window(self, x, t , end_t, window=1):
        num = 0
        for i in range(len(x)):
            d = end_t[i] - t[i]
            num = i
            if d <= (window * 12):
                break

        # print("num:", num)
        x = x[:num, :]
        t = t[:num]

        return x, t




    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i].float()
        mask = self.mask[i]
        time_stamp = self.time_stamp[i].float()
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



def get_dataloader(path="patient_data.csv", time_window=0.5):
    dataset = MySet(path=path, window=time_window)
    data_loader = DataLoader(
        dataset, batch_size=64, num_workers=1, shuffle=True
    )


    return data_loader

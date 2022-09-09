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
    def __init__(self, type="train", hid_dim=64):
        super(MySet, self).__init__()

        if type == "train":
            path = "nafld_series.pickle"
        else:
            path = "dm_2_series.pickle"


        with open(path, 'rb') as file:
             dict_get = pickle.load(file)
        print("OK")


        self.hid_dim = hid_dim
        self.id = dict_get["id"]

        self.x = dict_get["param"]
        self.x_forward = dict_get["param"]
        self.y = dict_get["label"]
        self.mask = dict_get["mask"]
        self.t = dict_get["time_stamp"]
        self.static_data = dict_get["static_data"]
        self.delta = dict_get["delta"]
        self.medication = dict_get["medication"]





    def __len__(self):
        return len(self.id)

    def __getitem__(self, i):
        id = self.id[i]
        h_mask = np.zeros((len(self.mask[id]), self.hid_dim))
        stamps = self.t[id]
        h_mask[:len(stamps), :] = 1
        static_data = torch.from_numpy(self.static_data[id]).float()

        x = torch.from_numpy(self.x[id]).float()
        y = torch.tensor(self.y[id]).float()
        delta = torch.from_numpy(self.delta[id]).float()
        mask = torch.from_numpy(self.mask[id]).float()
        x_forward = torch.from_numpy(self.x_forward[id]).float()
        h_mask = torch.from_numpy(h_mask).float()


        rec = {
            "x": x,
            "y": y,
            "delta": delta,
            "m": mask,
            "x_forward": x_forward,
            "h_mask": h_mask,
            "static_data": static_data

        }

        return rec




def collate_fn(recs):
    x = []
    y = []
    delta = []
    mask = []
    x_forward = []
    h_mask = []
    static_data = []
    for rec in recs:
        x.append(rec["x"])
        y.append(rec["y"])
        delta.append(rec["delta"])
        mask.append(rec["m"])
        x_forward.append(rec["x_forward"])
        h_mask.append(rec["h_mask"])
        static_data.append(rec["static_data"])

    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    delta = torch.stack(delta, dim=0)
    mask = torch.stack(mask, dim=0)
    x_forward = torch.stack(x_forward, dim=0)
    h_mask = torch.stack(h_mask, dim=0)
    static_data = torch.stack(static_data, dim=0)

    rec = {
        "x": x,
        "y": y,
        "delta": delta,
        "m": mask,
        "x_forward": x_forward,
        "h_mask": h_mask,
        "static_data": static_data
    }



    return rec

def get_loader(batch_size = 64, shuffle = True, type="train"):
    data_set = MySet(type=type)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
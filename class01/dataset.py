# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 11:17 上午
# @Author  : Ian
# @File    : dataset.py
# @Project : class01

import pandas as pd
import numpy as np
from config import config
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, train = True):
        super(TextDataset, self).__init__()
        if train:
            self.data_X = np.loadtxt(config.train_X)
            self.data_Y = np.loadtxt(config.train_Y)
        else:
            self.data_X = np.loadtxt(config.val_X)
            self.data_Y = np.loadtxt(config.val_Y)

    def __getitem__(self, index):
        x = self.data_X[index]
        y = np.zeros(1)
        y[0] = self.data_Y[index]
        x = torch.tensor(x, dtype=torch.int64)
        y = torch.tensor(y, dtype=torch.float32)
        return x,y

    def __len__(self):
        return len(self.data_X)


if __name__ == '__main__':

    d = TextDataset(train=True)
    print(d[0])
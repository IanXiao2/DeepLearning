# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 7:16 下午
# @Author  : Ian
# @File    : utils.py
# @Project : class01

import torch
from torch import nn
import time
from sklearn.metrics import accuracy_score
import numpy as np


def print_time_cost(since):
    time_cost = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_cost // 60, time_cost % 60)


def calc_acc(y_true, y_pred, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pred = y_pred.view(-1).cpu().detach().numpy() > threshold
    return accuracy_score(y_true, y_pred)

if __name__ == "__main__":
    print('-----------------')
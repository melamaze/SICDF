'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/Fed.py
'''

import copy
import torch
from torch import nn
import numpy as np

def FedAvg(w, weights):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weights[0]

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * weights[i]
    return w_avg


import torch
from torch.utils.data import TensorDataset, Dataset
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import pickle
from random import *
from torch.utils import data
import random
import math
from torch.utils import data
from torch.utils.data import TensorDataset
import tsaug

def Data_Augmentation(inputs, random_seed=None, theta_list=None, std_list=None):
    assert inputs.shape[1] == 2
    assert inputs.shape[2] == 128
    if random_seed == None:
        random_seed = randint(0, 8)
    if theta_list == None:
        theta_list = [0, math.pi/2, math.pi, math.pi/2*3]
    if std_list == None:
        std_list = [0, 0.05, 0.1, 0.15]
    if random_seed == 0:
        Data_Aug = inputs
    elif random_seed == 1 or random_seed == 2: # rotation
        theta = random.choice(theta_list)
        i_data = inputs[:, 0, :]
        q_data = inputs[:, 1, :]
        i_data_gen = math.cos(theta) * i_data - math.sin(theta) * q_data
        q_data_gen = math.sin(theta) * i_data + math.cos(theta) * q_data
        Data_Aug = torch.cat([i_data_gen.unsqueeze(1), q_data_gen.unsqueeze(1)], dim=1)
    elif random_seed == 3:
        flip_v = torch.zeros_like(inputs)
        flip_v[:, 0, :] = inputs[:, 0, :]
        flip_v[:, 1, :] = -inputs[:, 1, :]
        Data_Aug = flip_v
    elif random_seed == 4:
        flip_h = torch.zeros_like(inputs)
        flip_h[:, 0, :] = -inputs[:, 0, :]
        flip_h[:, 1, :] = inputs[:, 1, :]
        Data_Aug = flip_h
    elif random_seed == 5:
        flip_v_h = torch.zeros_like(inputs)
        flip_v_h[:, 0, :] = -inputs[:, 0, :]
        flip_v_h[:, 1, :] = -inputs[:, 1, :]
        Data_Aug = flip_v_h
    elif random_seed == 6:
        Data_Aug = torch.flip(inputs, dims=[2])
    elif random_seed == 7:
        Data_Aug = inputs
    elif random_seed == 8:
        std_temp = random.choice(std_list)
        noise = torch.normal(mean=0, std=std_temp, size=inputs.shape)
        Data_Aug = inputs + noise
    return Data_Aug

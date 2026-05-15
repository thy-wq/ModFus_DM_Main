import os
import torch
import numpy as np
import torch.nn as nn
import random
from torch.utils.data import Dataset
import pickle

def awgn(x, snr_db):
    """
    Adds complex AWGN to a signal.
    x: complex signal represented as a tensor of shape (2, N) for (I, Q)
    snr_db: desired SNR in dB
    """
    # Calculate signal power
    sig_power = torch.mean(x[0,:]**2 + x[1,:]**2)
    
    # Convert SNR from dB to linear scale
    snr_linear = 10.0 ** (snr_db / 10.0)
    
    # Calculate the required noise power
    noise_power = sig_power / snr_linear
    
    # The variance for each component (I and Q) is noise_power / 2.
    noise_std = torch.sqrt(noise_power / 2.0)
    
    # Generate noise
    noise = torch.randn_like(x) * noise_std
    
    return x + noise

class REAL_Filter(nn.Module):
    def __init__(self, args, train, snr):
        random.seed(1)
        self.args = args
        self.train = train  # training set or test set
        print("数据集使用的信噪比为", str(snr))
        print(type(snr))
        self.signal = []
        self.figure = []
        self.targets = []
        num_train = int(500)
        num_test = int(100)
        num_class = int(18)

        if self.train == True: # 训练
            data_root = os.path.join(r"E:\APG_1024\train.pt")
            data = torch.load(data_root).tensors[0]  # 所有类别的训练集
            label_per_snr = np.repeat(np.arange(num_class),num_train)
            self.targets = np.tile(label_per_snr, 18) # 前8800为0-10，依次重复20次
            self.data = data.reshape(-1,2,1024)
            print("全信噪比分布内+ 分布外类别load train data ", self.data.shape[0])
        else:
            data_root = os.path.join(r"E:\APG_1024\test.pt")
            data = torch.load(data_root).tensors[0]  # 所有类别的训练集
            label_per_snr = np.repeat(np.arange(num_class), num_test)
            self.targets = np.tile(label_per_snr, 18)  # 前8800为0-10，依次重复20次
            self.data = data.reshape(-1, 2, 1024)
            print("全信噪比分布内+ 分布外类别load test data ", self.data.shape[0])

        self.signal = np.array(data).reshape(-1,2,1024)
        self.targets = np.array(self.targets)
        self.signal = torch.as_tensor(np.array(self.signal), dtype=torch.float)
        # print("known:",np.unique(self.targets))
        print(f"train {train} ")
    def __getitem__(self, index):
        signal = self.signal[index].squeeze()
        # Add 10dB AWGN
        signal = awgn(signal, 10.0)
        AM = torch.sqrt(torch.square(signal).sum(dim=0))
        AM_max = torch.max(AM)
        data = signal / AM_max
        data = data.view(1, 2, 1024)
        return data, self.targets[index]

    def __len__(self):
        return len(self.signal)

class REAL_OSR(nn.Module):
    def __init__(self, args, known, snr=10, use_gpu=True, num_workers=0, batch_size=128):
        # print("调用 RML2016_OSR")
        np.random.seed(1)
        self.num_classes = len(known)
        self.known = known  # 已知未知的类别
        self.unknown = list(set(list(range(0, 11))) - set(known))

        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        trainset = REAL_Filter(args, train=True, snr=snr)
        # trainset.Filter(known=self.known)
        # print('All Train Data:', len(trainset))

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = REAL_Filter(args, train=False, snr=snr)
        # testset.Filter(known=self.known)
        print('All Test Data:', len(testset))

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = REAL_Filter(args, train=False, snr=snr)

        if self.unknown == []:
            self.out_loader = None
        else:
            self.out_loader = torch.utils.data.DataLoader(
                outset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )
        print('All OOD Data:', len(outset))

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test ID + OD: ', (len(testset) + len(outset)))
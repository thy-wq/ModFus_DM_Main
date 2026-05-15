import pickle
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import random
np.random.seed(1)

class RML2018(Dataset):
    def __init__(self, train, snr=10):
        random.seed(1)
        self.train = train  # training set or test set
        self.snr = snr
        self.root = 'F:\\python\\AMC_FSCIL\\data\\RML2018\\2018.01'
        self.data = []
        self.targets = []

        self.data_train = []
        self.data_test = []

        ratio = 0.7

        signal_path = 'D:\数据集\\2018.01\\RML2018_dB\\{}dB_X.pkl'.format(snr)
        label_path = 'D:\数据集\\2018.01\\RML2018_dB\\{}dB_Y.pkl'.format(snr)
        print("加载数据集的路径:", signal_path)
        with open(signal_path, "rb") as f:
            s = pickle.load(f)  # [24,4096,1024,2]
        with open(label_path, "rb") as f:
            l = pickle.load(f)  # [24,4096]
        classes = np.unique(l)
        num_class = int(classes.shape[0])
        num_sample_class = int(s.shape[0] * s.shape[1] / num_class) # 全数据集中每个类别每个信噪比 4096

        a_trian = int(ratio * num_sample_class)  # 每一类别用来训练的样本数目 2867
        b_test = int(num_sample_class - ratio * num_sample_class)  # 每类样本中用来测试的样本数目 1228

        class_list = list(range(0,num_class))
        for index in class_list:  # 按类别
            idnex_1 = list(range(num_sample_class))
            random.shuffle(idnex_1)
            for j in range(num_sample_class):
                if j < a_trian:
                    self.data_train.append(s[index, idnex_1[j], :, :])
                elif a_trian <= j < a_trian + b_test:
                    self.data_test.append(s[index, idnex_1[j], :, :])
        curent_class = len(class_list)

        if self.train:
            self.data = np.array(self.data_train)
            self.data = np.transpose(self.data, (0, 2, 1))
            self.targets = np.tile(np.arange(num_class)[:, np.newaxis], (1, a_trian)).astype(np.uint8)
            self.targets = self.targets.reshape(-1)
        else:
            self.data = np.array(self.data_test)
            self.data = np.transpose(self.data, (0, 2, 1))
            self.targets = np.tile(np.arange(num_class)[:, np.newaxis], (1, b_test)).astype(np.uint8)
            self.targets = self.targets.reshape(-1)
        self.data.astype(np.float32)

    def __getitem__(self, index):
        data = self.data[index]
        return data, self.targets[index]

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    rml = RML2018(train=True)
    print(rml.data.shape) # 2867 * 24 = 68808
    print(rml.targets.shape)

    rml = RML2018(train=False)
    print(rml.data.shape) # 1228 * 24 = 29472
    print(rml.targets.shape)
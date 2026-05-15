import pickle
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import random
np.random.seed(1)

def RML2016(snr=None): # 默认取的是0dB以上的
    xd = pickle.load(open("D:\数据集\RML2016.10a\\RML2016.10a_dict.pkl", 'rb'), encoding='iso-8859-1')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0])

    # del snrs[0:18]
    x = []
    lbl = []

    if snr == 100: # 仅高信噪比
        del snrs[0:10]
    elif snr == -100: # 全信噪比
        pass
    else:
        snrs = [snr]

    for mod in mods:
        for i in snrs:
            x.append(xd[(mod, i)])
            for i in range(xd[(mod, i)].shape[0]):
                lbl.append((mod, i))
    x = np.vstack(x)

    n_all_class = len(mods)
    n_per_class = np.array(x.shape[0] / n_all_class, dtype=np.int32)

    # x = x.reshape((-1, 256))
    # scaler = sklearn.preprocessing.MinMaxScaler()
    # x = scaler.fit_transform(x)

    x = x.reshape((n_all_class, n_per_class, 2, 128))

    # x = np.transpose(x, (0, 1, 2, 4, 3))

    # x1 = np.linspace(0,127,128)
    # a0 = y[9,19999,:,:,0].reshape(128,)
    # a1 = y[9,19999,:,:,1].reshape(128,)
    # plt.plot(x1, a0, color = 'red')
    # plt.plot(x1, a1, color = 'blue')

    return xd, snrs, mods, lbl, x, x.shape[0], x.shape[1]


class RML2016a(nn.Module):
    def __init__(self, train,snr=10):
        np.random.seed(1)
        self.train = train
        xd, snrs, mods, l, s, _, _ = RML2016(snr)
        ratio = 0.7
        classes = np.unique(mods)
        num_class = int(classes.shape[0])
        num_sample_class = int(s.shape[0] * s.shape[1] / num_class)
        a_trian = int(ratio * num_sample_class)  # 每一类别用来训练的样本数目
        b_test = int(num_sample_class - ratio * num_sample_class)  # 每类样本中用来测试的样本数目
        # print(a_trian,b_test) # 2867 1228
        data_train = []
        data_test = []
        for index in range(num_class):  # 按类别
            idnex_1 = list(range(num_sample_class))
            random.shuffle(idnex_1)
            for j in range(num_sample_class):
                if j < a_trian:
                    data_train.append(s[index, idnex_1[j], :, :])
                elif a_trian <= j < a_trian + b_test:
                    data_test.append(s[index, idnex_1[j], :, :])
        data_train = np.array(data_train).reshape(-1,2,1,128)
        # data_train = np.transpose(data_train, (0, 2, 1))
        data_train_label = np.tile(np.arange(num_class)[:, np.newaxis], (1, a_trian)).astype(np.uint8)
        data_train_label = data_train_label.reshape(-1)

        data_test = np.array(data_test).reshape(-1,2,1,128)
        data_test_label = np.tile(np.arange(num_class)[:, np.newaxis], (1, b_test)).astype(np.uint8)
        data_test_label = data_test_label.reshape(-1)
        # data_test = np.transpose(data_test, (0, 2, 1))

        if self.train:
            self.data = data_train
            self.targets = data_train_label
        else:
            self.data = data_test
            self.targets = data_test_label

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    rml = RML2016a(train=True)
    print(rml.data.shape)
    print(rml.targets.shape)

    rml = RML2016a(train=False)
    print(rml.data.shape)
    print(rml.targets.shape)
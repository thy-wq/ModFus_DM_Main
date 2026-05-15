import pickle
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import random
np.random.seed(1)

def RML2022(snr=None): # 默认取的是0dB以上的
    xd = pickle.load(open("D:\数据集\RML2022\\RML22.01A", 'rb'), encoding='latin1')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0])

    # del snrs[0:18]
    x = []
    lbl = []

    for mod in mods:
        for i in snrs:
            x.append(xd[(mod, i)])
            for i in range(xd[(mod, i)].shape[0]):
                lbl.append((mod, i))
    x = np.vstack(x) #

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


if __name__ == '__main__':
    RML2022()

    # rml = RML2016a(train=True)
    # print(rml.data.shape)
    # print(rml.targets.shape)
    #
    # rml = RML2016a(train=False)
    # print(rml.data.shape)
    # print(rml.targets.shape)
import os
import torch
import numpy as np
import torch.nn as nn
import random
from torch.utils.data import Dataset
import pickle


class RML2016a_Filter(nn.Module):
    def __init__(self, args, train, snr):
        self.args = args
        self.train = train  # training set or test set
        print("数据集使用的信噪比为", str(snr))
        print(type(snr))
        self.signal = []
        self.figure = []
        self.targets = []
        num_train = int(800)
        num_test = int(200)
        num_class = int(11)

        if snr == 100 or snr == "100":
            print("全信噪比数据")
            if self.train == True: # 训练
                data_root = os.path.join(r"E:\数据集\RML2016A_dB_Semi\train\100dB.pkl")
                with open(data_root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                label_per_snr = np.repeat(np.arange(num_class),num_train)
                self.targets = np.tile(label_per_snr, 20) # 前8800为0-10，依次重复20次
                self.data = data.reshape(-1,2,128)
                print("全信噪比分布内+ 分布外类别load train data ", self.data.shape[0])
            else:
                data_root = os.path.join(r"E:\数据集\RML2016A_dB_Semi\test\100dB.pkl")
                with open(data_root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                    f.close()
                label_per_snr = np.repeat(np.arange(num_class), num_test)
                self.targets = np.tile(label_per_snr, 20)  # 前8800为0-10，依次重复20次
                self.data = data.reshape(-1, 2, 128)
                print("全信噪比分布内+ 分布外类别load test data ", self.data.shape[0])
        elif snr == 1000 :
            print("0dB以上信号数据")
            if self.train == True: # 训练
                data_root = os.path.join(r"E:\数据集\RML2016A_dB_Semi\train\highsnr.pkl")
                with open(data_root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                label_per_snr = np.repeat(np.arange(num_class),num_train)
                self.targets = np.tile(label_per_snr, 10) # 前8800为0-10，依次重复20次
                self.data = data.reshape(-1,2,128)
                print("全信噪比分布内+ 分布外类别load train data ", self.data.shape[0])
            else:
                data_root = os.path.join(r"E:\数据集\RML2016A_dB_Semi\test\highsnr.pkl")
                with open(data_root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                    f.close()
                label_per_snr = np.repeat(np.arange(num_class), num_test)
                self.targets = np.tile(label_per_snr, 10)  # 前8800为0-10，依次重复20次
                self.data = data.reshape(-1, 2, 128)
                print("全信噪比分布内+ 分布外类别load test data ", self.data.shape[0])
        else:# print("单信噪比训练")
            print("单信噪比数据 snr = ", snr)
            if self.train == True:
                # root = os.path.join(r"D:\数据集\RML2016a_FS\RML2016a_old", str(snr) + "dB_old.pkl")
                root = os.path.join(r"E:\数据集\RML2016A_dB_Semi\train", str(snr) + "dB.pkl")
                with open(root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                    f.close()
                self.targets = np.tile(np.arange(num_class)[:, np.newaxis], (1, num_train)).astype(np.uint8)
                self.targets = self.targets.reshape(-1)
                self.num = num_train
                print("单信噪比分布内+ 分布外类别load train data ", data.shape[0])
            elif self.train == False:
                root = os.path.join(r"E:\数据集\RML2016A_dB_Semi\test", str(snr) + "dB.pkl")
                with open(root, 'rb') as f:
                    data = pickle.load(f)
                    f.close()
                self.targets = np.tile(np.arange(num_class)[:, np.newaxis], (1, num_test)).astype(np.uint8)
                self.targets = self.targets.reshape(-1)
                self.num = num_test
                print("单信噪比分布内+ 分布外类别load test data ", data.shape[0])

        self.signal = np.array(data).reshape(-1,2,128)
        self.targets = np.array(self.targets)
        if self.train and getattr(self.args, 'label_noise', 0.0) > 0:
            self.targets = self.apply_label_noise(self.targets, num_class)
        self.signal = torch.as_tensor(np.array(self.signal), dtype=torch.float)
        # print("known:",np.unique(self.targets))
        print(f"train {train} ")

    def apply_label_noise(self, targets, num_class):
        noise_ratio = float(getattr(self.args, 'label_noise', 0.0))
        noise_type = getattr(self.args, 'noise_type', 'symmetric')
        if noise_ratio <= 0 or noise_ratio >= 1:
            return targets

        targets = targets.copy()
        num_samples = len(targets)
        num_noisy = int(round(noise_ratio * num_samples))
        if num_noisy <= 0:
            return targets

        rng = np.random.default_rng()
        noisy_idx = rng.choice(num_samples, size=num_noisy, replace=False)

        for idx in noisy_idx:
            clean_label = int(targets[idx])
            if noise_type == 'asymmetric':
                new_label = (clean_label + 1) % num_class
            else:
                choices = list(range(num_class))
                choices.remove(clean_label)
                new_label = int(rng.choice(choices))
            targets[idx] = new_label

        print(f"Applied {noise_type} label noise: ratio={noise_ratio}, noisy_count={num_noisy}")
        return targets

    def __getitem__(self, index):
        signal = self.signal[index].squeeze()
        AM = torch.sqrt(torch.square(signal).sum(dim=0))
        AM_max = torch.max(AM)
        data = signal / AM_max
        data = data.view(1, 2, 128)
        return data, self.targets[index]

    def __len__(self):
        return len(self.signal)

class RML2016a_OSR(nn.Module):
    def __init__(self, args, known, snr=10, use_gpu=True, num_workers=0, batch_size=128):
        # print("调用 RML2016_OSR")
        self.num_classes = len(known)
        self.known = known  # 已知未知的类别
        self.unknown = list(set(list(range(0, 11))) - set(known))

        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        # trainset = RML2016_Filter(train=True)
        trainset = RML2016a_Filter(args, train=True, snr=snr)
        # trainset.Filter(known=self.known)
        # print('All Train Data:', len(trainset))

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = RML2016a_Filter(args, train=False, snr=snr)
        # testset.Filter(known=self.known)
        print('All Test Data:', len(testset))

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = RML2016a_Filter(args, train=False, snr=snr)

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


class RML2018_Filter(nn.Module):
    def __init__(self, train, snr):
        random.seed(1)
        self.train = train  # training set or test set

        self.signal = []
        self.figure = []
        self.targets = []
        num_train = int(3276)
        num_test = int(819)
        num_class = int(24)

        if self.train == True:
            print(f"load data from E:\\数据集\\RML2018_dB_Semi/{snr}dB_train.pkl")
            with open(f"E:\\数据集\\RML2018_dB_Semi\\{snr}dB_train.pkl", 'rb') as f:
                data = pickle.load(f)
                f.close()
            self.targets = np.tile(np.arange(num_class)[:, np.newaxis], (1, num_train)).astype(np.uint8)
            self.targets = self.targets.reshape(-1)
            self.num = num_train
            print("load train data ", data.shape[0])
        elif self.train == False:
            print(f"load data from E:\\数据集\\RML2018_dB_Semi/{snr}dB_test.pkl")
            with open(f"E:\\数据集\\RML2018_dB_Semi/{snr}dB_test.pkl", 'rb') as f:
                data = pickle.load(f)
                f.close()
            self.targets = np.tile(np.arange(num_class)[:, np.newaxis], (1, num_test)).astype(np.uint8)
            self.targets = self.targets.reshape(-1)
            self.num = num_test
            print("load test data ", data.shape[0])
        self.signal = np.array(data)
        self.targets = np.array(self.targets)
        self.signal = torch.as_tensor(np.array(self.signal), dtype=torch.float)


    def __getitem__(self, index):
        signal = self.signal[index].squeeze()
        # s = torch.square(signal)
        # print('I', signal[0,0],'Q',signal[1,0],'\t', s[0,0])
        AM = torch.sqrt(torch.square(signal).sum(dim=0))
        # print(AM[0])
        AM_max = torch.max(AM)
        # print(AM21[0,0],"\t",AM21[1,0],'\t',AM2[0])
        # print(AM2[0])
        data = signal / AM_max
        data = data.view(1, 2, 1024)
        return data,  self.targets[index]

    def __len__(self):
        return len(self.signal)


class RML2018_OSR(nn.Module):
    def __init__(self, known, use_gpu=True, num_workers=8, batch_size=128, snr=[24]):
        # print("调用 RML2016_OSR")
        self.num_classes = len(known)
        self.known = known  # 已知未知的类别

        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        # trainset = RML2016_Filter(train=True)
        trainset = RML2018_Filter(train=True, snr=snr)
        self.unknown = list(set(list(range(0, 24))) - set(known))
        print('unknown Labels: ', self.unknown)
        print('All Train Data:', len(trainset))

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = RML2018_Filter(train=False, snr=snr)
        print('All Test Data:', len(testset))

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = RML2018_Filter(train=False, snr=snr)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class RML2018v_Filter(nn.Module):
    def __init__(self, train, snr, length):
        random.seed(1)
        self.train = train  # training set or test set

        self.signal = []
        self.figure = []
        self.targets = []
        num_train = int(3276)
        num_test = int(819)
        num_class = int(24)
        self.length = length

        if self.train == True:
            print(f"load data from E:\\数据集\\RML2018_dB_Semi/{snr}dB_train.pkl")
            with open(f"E:\\数据集\\RML2018_dB_Semi\\{snr}dB_train.pkl", 'rb') as f:
                data = pickle.load(f)
                f.close()
            self.targets = np.tile(np.arange(num_class)[:, np.newaxis], (1, num_train)).astype(np.uint8)
            self.targets = self.targets.reshape(-1)
            self.num = num_train
            print("load train data ", data.shape[0])
        elif self.train == False:
            print(f"load data from E:\\数据集\\RML2018_dB_Semi/{snr}dB_test.pkl")
            with open(f"E:\\数据集\\RML2018_dB_Semi/{snr}dB_test.pkl", 'rb') as f:
                data = pickle.load(f)
                f.close()
            self.targets = np.tile(np.arange(num_class)[:, np.newaxis], (1, num_test)).astype(np.uint8)
            self.targets = self.targets.reshape(-1)
            self.num = num_test
            print("load test data ", data.shape[0])
        self.signal = np.array(data)
        self.targets = np.array(self.targets)
        self.signal = torch.as_tensor(np.array(self.signal), dtype=torch.float)


    def __getitem__(self, index):
        signal = self.signal[index].squeeze()
        # s = torch.square(signal)
        # print('I', signal[0,0],'Q',signal[1,0],'\t', s[0,0])
        AM = torch.sqrt(torch.square(signal).sum(dim=0))
        # print(AM[0])
        AM_max = torch.max(AM)
        # print(AM21[0,0],"\t",AM21[1,0],'\t',AM2[0])
        # print(AM2[0])
        data = signal / AM_max
        data = data.view(1, 2, 1024)
        data = random_crop(data, self.length)
        return data,  self.targets[index]

    def __len__(self):
        return len(self.signal)

# 随机截取长度为length信号
def random_crop(sequence, length):
    start = np.random.randint(0, 1024 - length)
    return sequence[:,:,start:start+length]

class RML2018v_OSR(nn.Module):
    def __init__(self, known,length, use_gpu=True, num_workers=8, batch_size=128, snr=[24]):
        # print("调用 RML2016_OSR")
        self.num_classes = len(known)
        self.known = known  # 已知未知的类别
        self.length = length

        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        # trainset = RML2016_Filter(train=True)
        trainset = RML2018v_Filter(train=True, snr=snr, length = self.length)
        self.unknown = list(set(list(range(0, 24))) - set(known))
        print('unknown Labels: ', self.unknown)
        print('All Train Data:', len(trainset))

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = RML2018v_Filter(train=False, snr=snr, length = self.length)
        print('All Test Data:', len(testset))

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = RML2018v_Filter(train=False, snr=snr, length = self.length)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class RML2022_Filter(nn.Module):
    def __init__(self, args, train, snr):
        random.seed(1)
        self.args = args
        self.train = train  # training set or test set
        print("数据集使用的信噪比为", str(snr))
        print(type(snr))
        self.signal = []
        self.figure = []
        self.targets = []
        num_train = int(1600)
        num_test = int(400)
        num_class = int(11)

        if snr == 100 or snr == "100":
            print("全信噪比训练")
            if self.train == True:
                data_root = os.path.join(r"E:\数据集\RML2022_DB_Semi\train\100dB.pkl")
                with open(data_root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                    f.close()
                label_per_snr = np.repeat(np.arange(num_class), num_train)
                self.targets = np.tile(label_per_snr, 21)  # 前8800为0-10，依次重复20次
                self.data = data.reshape(-1, 2, 128)
                print("全信噪比分布内+ 分布外类别load train data ", self.data.shape[0])
            else:
                data_root = os.path.join(r"E:\数据集\RML2022_DB_Semi\test\100dB.pkl")
                with open(data_root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                    f.close()
                label_per_snr = np.repeat(np.arange(num_class), num_test)
                self.targets = np.tile(label_per_snr, 21)  # 前8800为0-10，依次重复20次
                self.data = data.reshape(-1, 2, 128)
                print("全信噪比分布内+ 分布外类别load test data ", self.data.shape[0])
        elif snr == 1000:
            print("0dB以上信号训练")
            if self.train == True:
                data_root = os.path.join(r"E:\数据集\RML2022_DB_Semi\train\highsnr.pkl")
                with open(data_root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                    f.close()
                label_per_snr = np.repeat(np.arange(num_class), num_train)
                self.targets = np.tile(label_per_snr, 11)  # 前8800为0-10，依次重复20次
                self.data = data.reshape(-1, 2, 128)
                print("全信噪比分布内+ 分布外类别load train data ", self.data.shape[0])
            else:
                data_root = os.path.join(r"E:\数据集\RML2022_DB_Semi\test\highsnr.pkl")
                with open(data_root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                    f.close()
                label_per_snr = np.repeat(np.arange(num_class), num_test)
                self.targets = np.tile(label_per_snr, 11)  # 前8800为0-10，依次重复20次
                self.data = data.reshape(-1, 2, 128)
                print("全信噪比分布内+ 分布外类别load test data ", self.data.shape[0])
        elif snr == 1000 or snr == "100":
            print("全信噪比训练")
            if self.train == True:
                data_root = os.path.join(r"E:\数据集\RML2022_DB_Semi\train\100dB.pkl")
                with open(data_root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                    f.close()
                label_per_snr = np.repeat(np.arange(num_class), num_train)
                self.targets = np.tile(label_per_snr, 21)  # 前8800为0-10，依次重复20次
                self.data = data.reshape(-1, 2, 128)
                print("全信噪比分布内+ 分布外类别load train data ", self.data.shape[0])
            else:
                data_root = os.path.join(r"E:\数据集\RML2022_DB_Semi\test\100dB.pkl")
                with open(data_root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                    f.close()
                label_per_snr = np.repeat(np.arange(num_class), num_test)
                self.targets = np.tile(label_per_snr, 21)  # 前8800为0-10，依次重复20次
                self.data = data.reshape(-1, 2, 128)
                print("全信噪比分布内+ 分布外类别load test data ", self.data.shape[0])
        else:
            # print("单信噪比训练")
            print("snr = ", snr)
            if self.train == True:
                # root = os.path.join(r"D:\数据集\RML2016a_FS\RML2016a_old", str(snr) + "dB_old.pkl")
                root = os.path.join(r"E:\数据集\RML2022_DB_Semi\train", str(snr) + "dB.pkl")
                with open(root, 'rb') as f:
                    data = pickle.load(f)  # 所有类别的训练集
                    f.close()
                self.targets = np.tile(np.arange(num_class)[:, np.newaxis], (1, num_train)).astype(np.uint8)
                self.targets = self.targets.reshape(-1)
                self.num = num_train
                print("单信噪比分布内+ 分布外类别load train data ", data.shape[0])
            elif self.train == False:
                root = os.path.join(r"E:\数据集\RML2022_DB_Semi\test", str(snr) + "dB.pkl")
                with open(root, 'rb') as f:
                    data = pickle.load(f)
                    f.close()
                self.targets = np.tile(np.arange(num_class)[:, np.newaxis], (1, num_test)).astype(np.uint8)
                self.targets = self.targets.reshape(-1)
                self.num = num_test
                print("单信噪比分布内+ 分布外类别load test data ", data.shape[0])

        self.signal = np.array(data).reshape(-1,2,128)
        self.targets = np.array(self.targets)
        if self.train and getattr(self.args, 'label_noise', 0.0) > 0:
            self.targets = self.apply_label_noise(self.targets, num_class)
        self.signal = torch.as_tensor(np.array(self.signal), dtype=torch.float)
        # print("known:",np.unique(self.targets))
        print(f"train {train} ")

    def apply_label_noise(self, targets, num_class):
        noise_ratio = float(getattr(self.args, 'label_noise', 0.0))
        noise_type = getattr(self.args, 'noise_type', 'symmetric')
        if noise_ratio <= 0 or noise_ratio >= 1:
            return targets

        targets = targets.copy()
        num_samples = len(targets)
        num_noisy = int(round(noise_ratio * num_samples))
        if num_noisy <= 0:
            return targets

        rng = np.random.default_rng()
        noisy_idx = rng.choice(num_samples, size=num_noisy, replace=False)

        for idx in noisy_idx:
            clean_label = int(targets[idx])
            if noise_type == 'asymmetric':
                new_label = (clean_label + 1) % num_class
            else:
                choices = list(range(num_class))
                choices.remove(clean_label)
                new_label = int(rng.choice(choices))
            targets[idx] = new_label

        print(f"Applied {noise_type} label noise: ratio={noise_ratio}, noisy_count={num_noisy}")
        return targets

    def __getitem__(self, index):
        signal = self.signal[index].squeeze()
        AM = torch.sqrt(torch.square(signal).sum(dim=0))
        # print(AM[0])
        AM_max = torch.max(AM)
        data = signal / AM_max
        data = data.view(1, 2, 128)
        return data, self.targets[index]

    def __len__(self):
        return len(self.signal)

class RML2022_OSR(nn.Module):
    def __init__(self, args, known, snr=10, use_gpu=True, num_workers=0, batch_size=128):
        # print("调用 RML2016_OSR")
        np.random.seed(1)
        self.num_classes = len(known)
        self.known = known  # 已知未知的类别
        self.unknown = list(set(list(range(0, 11))) - set(known))

        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        # trainset = RML2016_Filter(train=True)
        trainset = RML2022_Filter(args, train=True, snr=snr)
        # trainset.Filter(known=self.known)
        # print('All Train Data:', len(trainset))

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = RML2022_Filter(args, train=False, snr=snr)
        # testset.Filter(known=self.known)
        print('All Test Data:', len(testset))

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = RML2022_Filter(args, train=False, snr=snr)

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


if __name__ == '__main__':
    Data = RML2018_OSR(known=[0, 1, 2], snr=24)
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    print('load 保存信号+时频图的结构体')
    for i, (signal, figure, label) in enumerate(testloader):
        print(signal.shape)
        print(figure.shape)
        break




import torch
import numpy as np
from torch.utils.data import Dataset

class PEmoDataset(Dataset):
    def __init__(self, path_train_data, path_train_data_cls_idx):

        self.train_data = np.load(path_train_data)
        self.train_x = self.train_data['x']  # shape(1052, 1024, 8)  1052 首歌, 1024 个时间步, 8 个特征
        self.train_y = self.train_data[
            'y']  # shape(1052, 1024, 8)  1052 首歌, 1024 个时间步, 8 个特征   train_y[x] = train_x[x+1]  是下一个时间步预测
        self.train_mask = self.train_data['mask']

        # 标签转换  区分四象限  1-4
        self.cls_idx = np.load(path_train_data_cls_idx)  # 四个类别的索引
        self.cls_1_idx = self.cls_idx['cls_1_idx']
        self.cls_2_idx = self.cls_idx['cls_2_idx']
        self.cls_3_idx = self.cls_idx['cls_3_idx']
        self.cls_4_idx = self.cls_idx['cls_4_idx']


        self.train_x = torch.from_numpy(self.train_x).long()  # np转tensor
        self.train_y = torch.from_numpy(self.train_y).long()
        self.train_mask = torch.from_numpy(self.train_mask).float()

        self.seq_len = self.train_x.shape[1]  # 序列长度 1024
        self.dim = self.train_x.shape[2]  # 维度 8

        print('train_x: ', self.train_x.shape)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index], self.train_mask[index]

    def __len__(self):
        return len(self.train_x)
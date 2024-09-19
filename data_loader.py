# 数据加载和处理
import numpy as np

import torch
from torch.utils.data import Dataset

class DataTrain(Dataset):
    def __init__(self, datas):
        self.x, self.y, self.SOCocv = datas

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data = self.x[index]
        SOC = self.y[index]
        SOCocv = self.SOCocv[index]

        return torch.tensor(data), torch.tensor(SOC), torch.tensor(SOCocv)

class DataTest(Dataset):
    def __init__(self, datas):
        self.x, self.y, self.k, self.SOCocv = datas

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        data = self.x[index]
        SOC = self.y[index]
        key = self.k[index][0]
        SOCocv = self.SOCocv[index]

        return torch.tensor(data), torch.tensor(SOC), key, torch.tensor(SOCocv)

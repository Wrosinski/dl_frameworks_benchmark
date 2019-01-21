import gc
import os

import numpy as np
import pandas as pd
import torch
from torch.utils import data


class BenchmarkDataset(data.Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

        self.divide = False
        self.is_test = False
        self.debug = False

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if index not in range(0, len(self.X)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        image = self.X[index]

        if self.divide:
            image = image / 255.
        image = torch.from_numpy(image).float().permute([2, 0, 1]).contiguous()
        if self.debug:
            print(image.shape)

        if not self.is_test:
            target = self.y[index]
            return image, target

        if self.is_test:
            return (image,)

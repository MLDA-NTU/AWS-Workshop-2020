import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMGSIZE = 28


__all__ = ['KannadaDataset']


class KannadaDataSet(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X.iloc[i,:]
        data = np.array(data).astype(np.uint8).reshape(IMGSIZE,IMGSIZE,1)
        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            # for train set, val set, and test set
            return (data, self.y[i])
        else:
            # for kaggle submission
            # since submission set will not have labels
            return data

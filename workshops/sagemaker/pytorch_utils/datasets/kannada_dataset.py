import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMGSIZE = 28


__all__ = ['KannadaDataset', ]


class KannadaDataset(Dataset):
    def __init__(self, images, targets=None, transforms=None):
        """
        """
        self.images = images    # numpy images
        self.targets = targets  # class labels array or None
        self.transforms = transforms

    def __len__(self):
        return (len(self.images))

    def __getitem__(self, i):
        data = self.images[i]
        data = np.array(data).astype(np.uint8).reshape(IMGSIZE,IMGSIZE,1)
        if self.transforms:
            data = self.transforms(data)

        if self.targets is not None:
            # for train set, val set, and test set
            return (data, self.targets[i])
        else:
            # for kaggle submission
            # since submission set will not have labels
            return data

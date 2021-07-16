import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'cifar-100-python', 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform
        self.classes = torch.unique(torch.tensor(self.data['coarse_labels'.encode()]))

    def __len__(self):
        return len(self.data['coarse_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['coarse_labels'.encode()][index]
        image = self.data['data'.encode()][index]

        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))
        image = torch.from_numpy(image)
        image=image.permute(2,0,1)

        image = F.to_pil_image(image)
#         image = transforms.ToPILImage(image)

        if self.transform:
            image = self.transform(image)
        return image, label, index

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'cifar-100-python', 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform
        self.classes = torch.unique(torch.tensor(self.data['coarse_labels'.encode()]))
        
    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['coarse_labels'.encode()][index]
        
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))
        image = torch.from_numpy(image)
        image=image.permute(2,0,1)
        image = F.to_pil_image(image)
#         image = transforms.ToPILImage(image)

        if self.transform:
            image = self.transform(image)
        return image, label, index

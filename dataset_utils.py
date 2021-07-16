import torch
from torch.utils.data import ConcatDataset, IterableDataset
from torchvision import transforms, datasets
import torch.nn as nn
import numpy as np
import random
import os
from PIL import Image
import cifar100_coarse_dataset
import bisect


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class STL10Instance(datasets.STL10):
    """STL10Instance Dataset.
    """
    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    
class ConcatInstance(ConcatDataset):
    """
    The input datasets here should be the Instance classes defined above. 
    #FIXME: assert the type of dataset passed as input
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
            
            assert isinstance(d, STL10Instance), "ConcatInstance only supports STL10Instance class"
            
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        image, target, index = self.datasets[dataset_idx][sample_idx]
        # pass original idx rather than dataset specific idx
        return image, target, idx
    

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class RandomApplyBlur(nn.Module):
    def __init__(self, kernel_size, p, sigma_min, sigma_max):
        
        super().__init__()
        self.kernel_size = kernel_size
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x):
        if random.random() > self.p:
            return x

        sigma = random.uniform(self.sigma_min, self.sigma_max)

        fn = filters.GaussianBlur2d(self.kernel_size, (sigma, sigma))
        return fn(x)


class MultiViewDataInjectorNCE(object):
    def __init__(self, image_size):

        transforms_1 = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        transforms_2 = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        self.transforms = [transforms_1, transforms_2]

    def __call__(self, sample):

        output = [transform(sample) for transform in self.transforms]
        return output



class MultiViewDataInjectorNCETwoViews(object):
    def __init__(self, image_size):

        transforms_1 = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        transforms_2 = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.38, 0.38, 0.38, 0.38),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        self.transforms = [transforms_1, transforms_2]

    def __call__(self, sample):

        output = [transform(sample) for transform in self.transforms]
        return output


class MultiViewDataInjectorNCEConsensus(object):
    def __init__(self, image_size, gaussian_blur_kernel_size=23):

        assert torch.__version__ >= '1.7.1'
        
        transforms_1 = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        transforms_2 = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomApply(
                torch.nn.ModuleList([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),]),
                p=0.8
            ),
            transforms.RandomApply(
                torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=gaussian_blur_kernel_size),]),
                p=1.0
            ),
            transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        ])

        self.transforms = [transforms_1, transforms_2]

    def __call__(self, sample):

        output = [transform(sample) for transform in self.transforms]
        return output


def get_data_loaders(datapath, image_size=224, batch_size=128, workers=8, get_train=False, eval_batch_size=256, nce_baseline=False, use_train_test=False, use_slightly_diff_views=False):

    if use_slightly_diff_views:
        train_transform = MultiViewDataInjectorNCETwoViews(image_size)

    elif nce_baseline:
        train_transform = MultiViewDataInjectorNCE(image_size)
    else:
        train_transform = MultiViewDataInjectorNCEConsensus(image_size)
    
    if 'STL10' in datapath:

        eval_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
                ])

        if not os.path.exists(datapath):
            os.makedirs(datapath)
    
        if not use_train_test:

            train_dataset = STL10Instance(datapath, split='train', download=True, transform=train_transform)
            train_dataset_for_eval = STL10Instance(datapath, split='train', download=True, transform=eval_transform)
            test_dataset = STL10Instance(datapath, split='test', download=True, transform=eval_transform)

        else:
            train_dataset = STL10Instance(datapath, split='train', download=True, transform=train_transform)
            test_dataset_temp = STL10Instance(datapath, split='test', download=True, transform=train_transform)
            # train_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset_temp])
            train_dataset = ConcatInstance([train_dataset, test_dataset_temp])

            train_dataset_for_eval = STL10Instance(datapath, split='train', download=True, transform=eval_transform)
            test_dataset = STL10Instance(datapath, split='test', download=True, transform=eval_transform)
            # train_dataset_for_eval = torch.utils.data.ConcatDataset([train_dataset_for_eval, test_dataset])
            train_dataset_for_eval = ConcatInstance([train_dataset_for_eval, test_dataset])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, sampler=None)

        train_loader_for_eval = torch.utils.data.DataLoader(
            train_dataset_for_eval, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, sampler=None)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, num_workers=workers,
            batch_size=batch_size, shuffle=True,
            pin_memory=True, sampler=None)

    elif 'CIFAR100' in datapath:

        eval_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
                ])
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        
        train_dataset = cifar100_coarse_dataset.CIFAR100Train(datapath, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

        train_dataset_for_eval = cifar100_coarse_dataset.CIFAR100Train(datapath, transform=eval_transform)
        train_loader_for_eval = torch.utils.data.DataLoader(train_dataset_for_eval, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

        test_dataset = cifar100_coarse_dataset.CIFAR100Test(datapath, transform=eval_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    elif 'CIFAR10' in datapath:

        #FIXME: what about normalization constants for different datasets?        
        eval_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
                ])
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        
        train_dataset = CIFAR10Instance(root=datapath, train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

        train_dataset_for_eval = CIFAR10Instance(root=datapath, train=True, download=True, transform=eval_transform)
        train_loader_for_eval = torch.utils.data.DataLoader(train_dataset_for_eval, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

        test_dataset = CIFAR10Instance(root=datapath, train=False, download=True, transform=eval_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    else:
        eval_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
                ])

        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'val')

        train_dataset = ImageFolderInstance(traindir,train_transform)
        train_dataset_for_eval = ImageFolderInstance(traindir, eval_transform)
        test_dataset = ImageFolderInstance(valdir,eval_transform)

        # FIXME: IS shuffle causing an issue?
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, sampler=None)

        train_loader_for_eval = torch.utils.data.DataLoader(
            train_dataset_for_eval, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, sampler=None)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, num_workers=workers,
            batch_size=batch_size, shuffle=False,
            pin_memory=True, sampler=None)
    
    if get_train:
        return train_loader, train_loader_for_eval, test_loader
    else:
        return train_loader_for_eval, test_loader


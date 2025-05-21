from cv2 import transform
from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms
from backbone.ResNetBottleneck import resnet50
from backbone.ResNetBlock import resnet18
from typing import Optional, Tuple
import torch.nn as nn
from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
import numpy as np
from PIL import Image

class ImageNet(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        if train:
            self.root = root + '/train'
        else:
            self.root = root + '/val'

        self.dataset = torchvision.datasets.ImageFolder(self.root, transform=None)
        self._data = np.array([path for path, _ in self.dataset.samples])
        self._targets = np.array([target for _, target in self.dataset.samples])

    @property
    def data(self):
        return self._data
    
    @property
    def targets(self):
        return self._targets
    
    @data.setter
    def data(self, data):
        self._data = data

    @targets.setter
    def targets(self, targets):
        self._targets = targets
        self.dataset.targets = targets
        self.dataset.samples = [(path, target) for path, target in zip(self._data, self._targets)]

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, self.logits[index]

        return img, target
    
    def __len__(self):
        return len(self.data)
    

class MyImageNet(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if train:
            self.root = root + '/train'
        else:
            self.root = root + '/val'

        self.dataset = torchvision.datasets.ImageFolder(self.root, transform=None)
        self._targets = np.array([target for _, target in self.dataset])
        self._data = np.array([path for path, _ in self.dataset.samples])

    @property
    def data(self):
        return self._data
    
    @property
    def targets(self):
        return self._targets
    
    @data.setter
    def data(self, data):
        self._data = data

    @targets.setter
    def targets(self, targets):
        self._targets = targets
        self.dataset.targets = targets
        self.dataset.samples = [(path, target) for path, target in zip(self._data, self._targets)]

    def __getitem__(self, index):
        img, target = self.dataset[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        original_img = img.copy()
        original_img = self.not_aug_transform(original_img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, original_img
    
    def __len__(self):
        return len(self.dataset)
    

class SequentialImagenet(ContinualDataset):
    """Sequential version of the ImageNet dataset."""

    NAME = 'seq-imagenet'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 100
    N_TASKS = 10
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    SIZE = (224, 224)
    TRANSFORM = transforms.Compose(
        [transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
    
    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(self.MEAN, self.STD)])

        train_dataset = MyImageNet(base_path() + 'imagenet',
                                   train=True, download=True, transform=transform)
        test_dataset = ImageNet(base_path() + 'imagenet',
                                  train=False, download=True, transform=test_transform)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test
    
    @staticmethod
    def get_backbone():
        return resnet18(nclasses=SequentialImagenet.N_CLASSES, cifar=False)
    
    @staticmethod
    def get_loss():
        return torch.nn.CrossEntropyLoss()
    
    def get_transform(self):
        transform = transforms.Compose([transforms.ToPILImage(), self.TRANSFORM])
        return transform
    
    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialImagenet.MEAN, SequentialImagenet.STD)
    
    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialImagenet.MEAN, SequentialImagenet.STD)
        return transform
    
    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 60
    
    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128



class ImageNet100(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        if train:
            self.root = root + '/train'
        else:
            self.root = root + '/val'


        self.dataset = torchvision.datasets.ImageFolder(self.root, transform=None)
        self._data = np.array([path for path, _ in self.dataset.samples])
        self._targets = np.array([target for _, target in self.dataset.samples])

    @property
    def data(self):
        return self._data
    
    @property
    def targets(self):
        return self._targets

    @data.setter
    def data(self, data):
        self._data = data

    @targets.setter
    def targets(self, targets):
        self._targets = targets
        self.dataset.targets = targets
        self.dataset.samples = [(path, target) for path, target in zip(self._data, self._targets)]

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, self.logits[index]

        return img, target

    def __len__(self):
        return len(self.data)
    
    
class MyImageNet100(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if train:
            self.root = root + '/train'
        else:
            self.root = root + '/val'

        self.dataset = torchvision.datasets.ImageFolder(self.root, transform=None)
        self._targets = np.array([target for _, target in self.dataset])
        self._data = np.array([path for path, _ in self.dataset.samples])
        
    @property
    def data(self):
        return self._data
    
    @property
    def targets(self):
        return self._targets
    
    @data.setter
    def data(self, data):
        self._data = data

    @targets.setter
    def targets(self, targets):
        self._targets = targets
        self.dataset.targets = targets
        self.dataset.samples = [(path, target) for path, target in zip(self._data, self._targets)]

    def __getitem__(self, index):
        img, target = self.dataset[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        original_img = img.copy()
        original_img = self.not_aug_transform(original_img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, original_img

    def __len__(self):
        return len(self.dataset)

class SequentialImagenet100(ContinualDataset):
    """Sequential version of the ImageNet-100 dataset."""

    NAME = 'seq-imagenet100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    SIZE = (224, 224)
    TRANSFORM = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(self.MEAN, self.STD)])

        train_dataset = MyImageNet100(base_path() + 'imagenet100',
                                   train=True, download=True, transform=transform)
        test_dataset = ImageNet100(base_path() + 'imagenet100',
                                  train=False, download=True, transform=test_transform)
        print("before")
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test
    
    @staticmethod
    def get_backbone():
        return resnet50(num_classes=SequentialImagenet100.N_CLASSES, pretrained=False)
    
    @staticmethod
    def get_loss():
        return torch.nn.CrossEntropyLoss()
    
    def get_transform(self):
        transform = transforms.Compose([transforms.ToPILImage(), self.TRANSFORM])
        return transform
    
    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialImagenet100.MEAN, SequentialImagenet100.STD)
    
    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialImagenet100.MEAN, SequentialImagenet100.STD)
        return transform
    
    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50
    
    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128
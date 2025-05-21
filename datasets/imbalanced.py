from backbone.ResNetBlock import resnet18
from datasets.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.seq_cifar10 import base_path
from typing import Tuple
import torch
import torch.nn.functional as F
from datasets.utils import set_default_from_args


class ImbalancedCIFAR10(IMBALANCECIFAR10):

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, self.logits[index]

        return img, target
    
class MyImbalancedCIFAR10(IMBALANCECIFAR10):
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                    transform=None, target_transform=None,
                    download=False):
        super(MyImbalancedCIFAR10, self).__init__(root, imb_type, imb_factor, rand_number, train, transform, target_transform, download)
        
        self.not_aug_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        org_img = img.copy()
        org_img = self.not_aug_transform(org_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, org_img, self.logits[index]

        return img, target, org_img
    
class SequentialImbalancedCIFAR10(ContinualDataset):
    """Sequential CIFAR10 Dataset.
    Args:
        root: root directory where the dataset will be stored
        train: if True, creates the dataset from training set, otherwise from
            test set
        transform: a function/transform that takes in an PIL image and returns a
            transformed version
        target_transform: a function/transform that takes in the target and
            transforms it
        download: if true, downloads the dataset from the internet and puts it in
            root directory. If dataset is already downloaded, it is not
            downloaded again.
        batch_size: size of the batch
        kwargs: additional arguments of the parent class
    """

    NAME = 'seq_imbalanced_cifar10'
    SETTING = 'class-il'
    N_CLASSES = 10
    N_TASKS = 10
    N_CLASSES_PER_TASK = 1
    SIZE = (32, 32)
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Class method that returns the train and test loaders."""
        transform = self.TRANSFORM

        imb_factor = self.args.imb_ratio
        rand_number = self.args.seed

        train_dataset = MyImbalancedCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform, imb_factor=imb_factor, rand_number=rand_number)
        test_dataset = ImbalancedCIFAR10(base_path() + 'CIFAR10', train=False,
                                  download=True, transform=self.TEST_TRANSFORM, imb_factor=imb_factor, rand_number=rand_number)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test
    
    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImbalancedCIFAR10.TRANSFORM])
        return transform
    
    @staticmethod
    def get_backbone():
        return resnet18(SequentialImbalancedCIFAR10.N_CLASSES_PER_TASK
                        * SequentialImbalancedCIFAR10.N_TASKS)
    
    @staticmethod
    def get_loss():
        return F.cross_entropy
    
    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialImbalancedCIFAR10.MEAN, SequentialImbalancedCIFAR10.STD)
        return transform
    
    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialImbalancedCIFAR10.MEAN, SequentialImbalancedCIFAR10.STD)
        return transform
    
    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50
    
    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32
    
class ImbalancedCIFAR100(IMBALANCECIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, self.logits[index]

        return img, target
    
class MyImbalancedCIFAR100(IMBALANCECIFAR100):
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                    transform=None, target_transform=None,
                    download=False):
        super(MyImbalancedCIFAR100, self).__init__(root, imb_type, imb_factor, rand_number, train, transform, target_transform, download)
        
        self.not_aug_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        org_img = img.copy()
        org_img = self.not_aug_transform(org_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, org_img, self.logits[index]

        return img, target, org_img

 
class SequentialImbalancedCIFAR100(ContinualDataset):
    """Sequential CIFAR100 Dataset.
    Args:
        root: root directory where the dataset will be stored
        train: if True, creates the dataset from training set, otherwise from
            test set
        transform: a function/transform that takes in an PIL image and returns a
            transformed version
        target_transform: a function/transform that takes in the target and
            transforms it
        download: if true, downloads the dataset from the internet and puts it in
            root directory. If dataset is already downloaded, it is not
            downloaded again.
        batch_size: size of the batch
        kwargs: additional arguments of the parent class
    """

    NAME = 'seq_imbalanced_cifar100'
    SETTING = 'class-il'
    N_CLASSES = 100
    N_TASKS = 10
    N_CLASSES_PER_TASK = 10
    SIZE = (32, 32)
    MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    
    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Class method that returns the train and test loaders."""
        transform = self.TRANSFORM

        imb_factor = self.args.imb_ratio
        rand_number = self.args.seed

        train_dataset = MyImbalancedCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform, imb_factor=imb_factor, rand_number=rand_number)
        test_dataset = ImbalancedCIFAR100(base_path() + 'CIFAR100', train=False,
                                  download=True, transform=self.TEST_TRANSFORM, imb_factor=imb_factor, rand_number=rand_number)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test
    
    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImbalancedCIFAR100.TRANSFORM])
        return transform
    
    @staticmethod
    def get_backbone():
        return resnet18(SequentialImbalancedCIFAR100.N_CLASSES_PER_TASK
                        * SequentialImbalancedCIFAR100.N_TASKS)
    
    @staticmethod
    def get_loss():
        return F.cross_entropy
    
    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialImbalancedCIFAR100.MEAN, SequentialImbalancedCIFAR100.STD)
        return transform
    
    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialImbalancedCIFAR100.MEAN, SequentialImbalancedCIFAR100.STD)
        return transform
    
    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50
    
    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32
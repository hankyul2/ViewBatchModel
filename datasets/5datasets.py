
import random
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Subset, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from timm.data import create_transform

from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive
from datasets.utils import set_default_from_args
from datasets.seq_tinyimagenet import base_path
from datasets.transforms.denormalization import DeNormalize
from backbone.ResNetBlock import resnet18
from PIL import Image
from dataset_utils import read_image_file, read_label_file

import os
from typing import Any, Callable, Optional, Tuple
import utils

class MNIST_RGB(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST_RGB, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()
        self.data = self.data.numpy()
        self.targets = self.targets.numpy()
        tmp = []
        for i, img in enumerate(self.data):
            image = Image.fromarray(img, mode='L').convert('RGB')
            tmp.append(np.array(image))

        self.data = np.array(tmp)

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FashionMNIST(MNIST_RGB):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class NotMNIST(MNIST_RGB):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip'
        self.filename = 'notMNIST.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()

        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')

        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')


        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img)
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3,  0, 1, 2))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class MyFiveDatasets(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, datasets=[MNIST_RGB, FashionMNIST, NotMNIST, CIFAR10, SVHN]):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.train = train

        self.data = []
        self.targets = []

        for i, dataset in enumerate(datasets):
            print(f"Loading {dataset.__name__}")
            if dataset.__name__ == 'SVHN':
                dset = dataset(root=root, split='train' if train else 'test', transform=None, target_transform=None, download=download)
                for image in dset.data:
                    self.data.append(cv2.resize(image, (32, 32)))
                target = (i + 1) * np.array(dset.targets)
                self.targets.extend(target)
            else:
                dset = dataset(root=root, train=train, transform=None, target_transform=None, download=download)
                for image in dset.data:
                    self.data.append(cv2.resize(image, (32, 32)))
                target = (i + 1) * np.array(dset.targets)
                self.targets.extend(target)

        print(len(self.data))

        # for i, img in enumerate(self.data):
        #     image = cv2.resize(img, (32, 32))
        #     self.data[i] = image

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

    def __len__(self) -> int:
        return len(self.data)


class FiveDatasets(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, datasets=[MNIST_RGB, FashionMNIST, NotMNIST, CIFAR10, SVHN]):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.data = []
        self.targets = []

        for i, dataset in enumerate(datasets):
            print(f"Loading {dataset.__name__}")
            if dataset.__name__ == 'SVHN':
                dset = dataset(root=root, split='train' if train else 'test', transform=None, target_transform=None, download=download)
                for image in dset.data:
                    self.data.append(cv2.resize(image, (32, 32)))
                target = (i + 1) * np.array(dset.targets)
                self.targets.extend(target)
            else:
                dset = dataset(root=root, train=train, transform=None, target_transform=None, download=download)
                for image in dset.data:
                    self.data.append(cv2.resize(image, (32, 32)))
                target = (i + 1) * np.array(dset.targets)
                self.targets.extend(target)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img, mode='RGB')

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, self.logits[index]

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class SeqFiveDatasets(ContinualDataset):
    """Sequential CIFAR10 Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'seq-five-datasets'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]

    TRANSFORM = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    def __init__(self, args):
        super(SeqFiveDatasets, self).__init__(args)

        self.datasets = [MNIST_RGB, FashionMNIST, NotMNIST, CIFAR10, SVHN]
        random.seed(args.seed)
        random.shuffle(self.datasets)

    def get_data_loaders(self):
        transform = self.TRANSFORM
        transform_test = self.TEST_TRANSFORM

        # datasets = [MNIST, FashionMNIST, NotMNIST, CIFAR10, SVHN]
        # random.seed(self.args.seed)
        # random.shuffle(datasets)

        train_dataset = MyFiveDatasets(base_path() + 'five_datasets', train=True, transform=transform, datasets=self.datasets)
        test_dataset = FiveDatasets(base_path() + 'five_datasets', train=False, transform=transform_test, datasets=self.datasets)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test
        
    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SeqFiveDatasets.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SeqFiveDatasets.N_CLASSES_PER_TASK
                        * SeqFiveDatasets.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SeqFiveDatasets.MEAN, SeqFiveDatasets.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SeqFiveDatasets.MEAN, SeqFiveDatasets.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32
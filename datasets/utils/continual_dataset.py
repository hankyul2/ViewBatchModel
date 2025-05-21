# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import math
from random import randint, random, shuffle
import sys
from typing import Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as scheds
from torch.utils.data import DataLoader, Dataset

from datasets.utils.validation import get_validation_indexes
from utils.conf import create_seeded_dataloader
from datasets.utils import DEFAULT_ARGS
from torch.utils.data import Sampler


class ViewBatchReplay(Sampler):
    """
    View-Batch Replay includes multiple views of sample in a batch.
    `num_repeats` corresponding to the number of relearning intervals in our paper.
    This implementation is borrowed from the TIMM library.
    """

    def __init__(
            self,
            dataset_len,
            num_replicas=None,
            rank=None,
            shuffle=True,
            num_repeats=3,
            selected_round=256,
            selected_ratio=0,
            no_buffer=False,
            only_buffer=False,
    ):
        if num_replicas is None:
            num_replicas = 1

        if rank is None:
            rank = 0

        self.dataset_len = dataset_len
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.selected_ratio = selected_ratio
        self.selected_round = selected_round
        self.no_buffer = no_buffer
        self.only_buffer = only_buffer

        self.set_num_repeats_and_others(self.num_repeats)
        print(f'no_buffer: {self.no_buffer}, only_buffer: {self.only_buffer}')

    def set_num_repeats_and_others(self, num_repeats):
        self.num_repeats = num_repeats
        self.num_samples = int(
            math.ceil(self.dataset_len * num_repeats / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        # Determine the number of samples to select per epoch for each rank.
        # num_selected logic defaults to be the same as original RASampler impl, but this one can be tweaked
        # via selected_ratio and selected_round args.
        selected_ratio = (
                self.selected_ratio or self.num_replicas
        )  # ratio to reduce selected samples by, num_replicas if 0
        if self.selected_round:
            self.num_selected_samples = int(
                math.floor(
                    self.dataset_len
                    // self.selected_round
                    * self.selected_round
                    / selected_ratio
                )
            )
        else:
            self.num_selected_samples = int(
                math.ceil(self.dataset_len / selected_ratio)
            )

    def set_dataset_len(self, dataset_len):
        self.dataset_len = dataset_len
        self.set_num_repeats_and_others(self.num_repeats)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch // self.num_repeats)
        if self.shuffle:
            if self.no_buffer or self.only_buffer:
                indices = torch.randperm(self.dataset_len - 200, generator=g)
            else:
                indices = torch.randperm(self.dataset_len, generator=g)
        else:
            indices = torch.arange(start=0, end=self.dataset_len)

        if self.only_buffer:  # 200 is for the buffer size.
            buffer_indices = torch.tensor(list(range(self.dataset_len - 200, self.dataset_len)))
            buffer_indices = buffer_indices[torch.randperm(200, generator=g)]
            buffer_indices = torch.repeat_interleave(buffer_indices, repeats=self.num_repeats, dim=0).tolist()
            indices = buffer_indices + indices.tolist()
        else:
            indices = torch.repeat_interleave(
                indices, repeats=self.num_repeats, dim=0
            ).tolist()

        if self.no_buffer:
            buffer_indices = list(range(self.dataset_len - 200, self.dataset_len))
            shuffle(buffer_indices)
            indices = buffer_indices + indices

        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        print(f"len(indices): {len(indices)}, total_size: {self.total_size}")
        assert len(indices) == self.total_size or self.only_buffer

        # subsample per rank
        previous_sample_size = (
                self.num_selected_samples
                * self.num_replicas
                * (self.epoch % self.num_repeats)
        )
        indices = indices[previous_sample_size + self.rank: self.total_size: self.num_replicas]

        # return up to num selected samples
        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ContinualDataset(object):
    """
    A base class for defining continual learning datasets.

    Attributes:
        NAME (str): the name of the dataset
        SETTING (str): the setting of the dataset
        N_CLASSES_PER_TASK (int): the number of classes per task
        N_TASKS (int): the number of tasks
        N_CLASSES (int): the number of classes
        SIZE (Tuple[int]): the size of the dataset
        train_loader (DataLoader): the training loader
        test_loaders (List[DataLoader]): the test loaders
        i (int): the current task
        c_task (int): the current task
        args (Namespace): the arguments which contains the hyperparameters
    """

    NAME: str
    SETTING: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int
    N_CLASSES: int
    SIZE: Tuple[int]
    AVAIL_SCHEDS = ["multisteplr"]

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.

        Args:
            args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.c_task = -1
        self.args = args
        if self.SETTING == "class-il":
            self.N_CLASSES = (
                self.N_CLASSES
                if hasattr(self, "N_CLASSES")
                else (
                    (self.N_CLASSES_PER_TASK * self.N_TASKS)
                    if isinstance(self.N_CLASSES_PER_TASK, int)
                    else sum(self.N_CLASSES_PER_TASK)
                )
            )
        else:
            self.N_CLASSES = self.N_CLASSES_PER_TASK

        if self.args.permute_classes:
            if not hasattr(self.args, "class_order"):  # set only once
                if self.args.seed is not None:
                    np.random.seed(self.args.seed)
                if isinstance(self.N_CLASSES_PER_TASK, int):
                    self.args.class_order = np.random.permutation(
                        self.N_CLASSES_PER_TASK * self.N_TASKS
                    )
                else:
                    self.args.class_order = np.random.permutation(
                        sum(self.N_CLASSES_PER_TASK)
                    )

        if self.args.validation:
            self._c_seed = (
                self.args.seed if self.args.seed is not None else torch.initial_seed()
            )

        if args.joint:
            self.N_CLASSES_PER_TASK = self.N_CLASSES
            self.N_TASKS = 1

        if not all(
                (
                        self.NAME,
                        self.SETTING,
                        self.N_CLASSES_PER_TASK,
                        self.N_TASKS,
                        self.SIZE,
                        self.N_CLASSES,
                )
        ):
            raise NotImplementedError(
                "The dataset must be initialized with all the required fields."
            )

        self.aug_repeat = args.aug_repeat

    def update_default_args(self):
        """
        Updates the default arguments with the ones specified in the dataset class.
        Default arguments are defined in the DEFAULT_ARGS dictionary and set by the 'set_default_from_args' decorator.

        Returns:
            Namespace: the updated arguments
        """

        if self.args.dataset not in DEFAULT_ARGS:  # no default args for this dataset
            return self.args

        for k, v in DEFAULT_ARGS[self.args.dataset].items():
            assert hasattr(
                self.args, k
            ), f"Argument {k} set by the `set_default_from_args` decorator is not present in the arguments."

            if getattr(self.args, k) is None:
                setattr(self.args, k, v)
            else:
                if getattr(self.args, k) != v:
                    print(
                        "Warning: {} set to {} instead of {}.".format(
                            k, getattr(self.args, k), v
                        ),
                        file=sys.stderr,
                    )

        return self.args

    def get_offsets(self, task_idx: int = None):
        """
        Compute the start and end class index for the current task.

        Args:
            task_idx (int): the task index

        Returns:
            tuple: the start and end class index for the current task
        """
        if self.SETTING == "class-il" or self.SETTING == "task-il":
            task_idx = task_idx if task_idx is not None else self.c_task
        else:
            task_idx = 0

        start_c = (
            self.N_CLASSES_PER_TASK * task_idx
            if isinstance(self.N_CLASSES_PER_TASK, int)
            else sum(self.N_CLASSES_PER_TASK[:task_idx])
        )
        end_c = (
            self.N_CLASSES_PER_TASK * (task_idx + 1)
            if isinstance(self.N_CLASSES_PER_TASK, int)
            else sum(self.N_CLASSES_PER_TASK[: task_idx + 1])
        )

        return start_c, end_c

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        raise NotImplementedError

    @staticmethod
    def get_backbone() -> nn.Module:
        """Returns the backbone to be used for the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_transform() -> nn.Module:
        """Returns the transform to be used for the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_loss() -> nn.Module:
        """Returns the loss to be used for the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """Returns the transform used for normalizing the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """Returns the transform used for denormalizing the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_scheduler(
            model, args: Namespace, reload_optim=True
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler to be used for the current dataset.
        If `reload_optim` is True, the optimizer is reloaded from the model. This should be done at least ONCE every task
        to ensure that the learning rate is reset to the initial value.
        """
        if args.lr_scheduler is not None:
            if reload_optim or not hasattr(model, "opt"):
                model.opt = model.get_optimizer()
            # check if lr_scheduler is in torch.optim.lr_scheduler
            supported_scheds = {
                sched_name.lower(): sched_name
                for sched_name in dir(scheds)
                if sched_name.lower() in ContinualDataset.AVAIL_SCHEDS
            }
            sched = None
            if args.lr_scheduler.lower() in supported_scheds:
                if args.lr_scheduler.lower() == "multisteplr":
                    assert (
                            args.lr_milestones is not None
                    ), "MultiStepLR requires `--lr_milestones`"
                    sched = getattr(
                        scheds, supported_scheds[args.lr_scheduler.lower()]
                    )(
                        model.opt,
                        milestones=args.lr_milestones,
                        gamma=args.sched_multistep_lr_gamma,
                    )

            if sched is None:
                raise ValueError("Unknown scheduler: {}".format(args.lr_scheduler))
            return sched
        return None

    def get_iters(self):
        """Returns the number of iterations to be used for the current dataset."""
        raise NotImplementedError(
            "The dataset does not implement the method `get_iters` to set the default number of iterations."
        )

    def get_epochs(self):
        """Returns the number of epochs to be used for the current dataset."""
        raise NotImplementedError(
            "The dataset does not implement the method `get_epochs` to set the default number of epochs."
        )

    def get_batch_size(self):
        """Returns the batch size to be used for the current dataset."""
        raise NotImplementedError(
            "The dataset does not implement the method `get_batch_size` to set the default batch size."
        )

    def get_minibatch_size(self):
        """Returns the minibatch size to be used for the current dataset."""
        return self.get_batch_size()


def _get_mask_unlabeled(train_dataset, setting: ContinualDataset):
    if setting.args.label_perc == 1:
        return np.zeros(train_dataset.targets.shape[0]).astype("bool")
    else:
        lpc = int(
            setting.args.label_perc
            * (train_dataset.targets.shape[0] // setting.N_CLASSES_PER_TASK)
        )
        ind = np.indices(train_dataset.targets.shape)[0]
        mask = []
        for i_label, _ in enumerate(np.unique(train_dataset.targets)):
            partial_targets = train_dataset.targets[train_dataset.targets == i_label]
            current_mask = np.random.choice(
                partial_targets.shape[0],
                max(partial_targets.shape[0] - lpc, 0),
                replace=False,
            )

            mask = np.append(mask, ind[train_dataset.targets == i_label][current_mask])

        return mask.astype(np.int32)


def _prepare_data_loaders(train_dataset, test_dataset, setting: ContinualDataset):
    if (
            isinstance(train_dataset.targets, list)
            or not train_dataset.targets.dtype is torch.long
    ):
        train_dataset.targets = torch.tensor(train_dataset.targets, dtype=torch.long)
    if (
            isinstance(test_dataset.targets, list)
            or not test_dataset.targets.dtype is torch.long
    ):
        test_dataset.targets = torch.tensor(test_dataset.targets, dtype=torch.long)

    setting.unlabeled_mask = _get_mask_unlabeled(train_dataset, setting)

    if setting.unlabeled_mask.sum() != 0:
        train_dataset.targets[setting.unlabeled_mask] = -1  # -1 is the unlabeled class

    return train_dataset, test_dataset


def store_masked_loaders(
        train_dataset: Dataset, test_dataset: Dataset, setting: ContinualDataset
) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.

    Attributes:
        train_dataset (Dataset): the training dataset
        test_dataset (Dataset): the test dataset
        setting (ContinualDataset): the setting of the dataset

    Returns:
        the training and test loaders
    """
    if not isinstance(train_dataset.targets, np.ndarray):
        train_dataset.targets = np.array(train_dataset.targets)
    if not isinstance(test_dataset.targets, np.ndarray):
        test_dataset.targets = np.array(test_dataset.targets)

    if setting.args.permute_classes:
        train_dataset.targets = setting.args.class_order[train_dataset.targets]
        test_dataset.targets = setting.args.class_order[test_dataset.targets]

    if setting.args.validation:
        train_idxs, val_idxs = get_validation_indexes(
            setting.args.validation, train_dataset, setting.args.seed
        )

        test_dataset.data = train_dataset.data[val_idxs]
        test_dataset.targets = train_dataset.targets[val_idxs]

        train_dataset.data = train_dataset.data[train_idxs]
        train_dataset.targets = train_dataset.targets[train_idxs]

    if setting.SETTING == "class-il" or setting.SETTING == "task-il":
        train_mask = np.logical_and(
            train_dataset.targets >= setting.i,
            train_dataset.targets < setting.i + setting.N_CLASSES_PER_TASK,
        )

        if setting.args.validation_mode == "current":
            test_mask = np.logical_and(
                test_dataset.targets >= setting.i,
                test_dataset.targets < setting.i + setting.N_CLASSES_PER_TASK,
            )
        elif setting.args.validation_mode == "complete":
            test_mask = np.logical_and(
                test_dataset.targets >= 0,
                test_dataset.targets < setting.i + setting.N_CLASSES_PER_TASK,
            )
        else:
            raise ValueError(
                "Unknown validation mode: {}".format(setting.args.validation_mode)
            )

        test_dataset.data = test_dataset.data[test_mask]
        test_dataset.targets = test_dataset.targets[test_mask]

        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = train_dataset.targets[train_mask]

    train_dataset, test_dataset = _prepare_data_loaders(
        train_dataset, test_dataset, setting
    )

    shuffle = True
    sampler = None
    if setting.aug_repeat > 1:
        sampler = ViewBatchReplay(len(train_dataset), num_repeats=setting.aug_repeat, no_buffer=setting.args.no_buffer,
                                  only_buffer=setting.args.only_buffer)
        shuffle = False

    train_loader = create_seeded_dataloader(
        setting.args, train_dataset, batch_size=setting.args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=setting.args.num_workers
    )
    test_loader = create_seeded_dataloader(
        setting.args, test_dataset, batch_size=setting.args.batch_size, shuffle=False,
        num_workers=setting.args.num_workers
    )
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    if setting.SETTING == "task-il" or setting.SETTING == "class-il":
        setting.i += setting.N_CLASSES_PER_TASK
        setting.c_task += 1

    if setting.args.dataset == 'seq-domainnet' or setting.args.dataset == 'seq-domainnet-128':
        setting.c_task += 1

    return train_loader, test_loader

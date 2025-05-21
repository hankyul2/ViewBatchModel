# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
import torch.nn.functional as F
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, fill_buffer, icarl_replay


class ICarl(ContinualModel):
    NAME = 'icarl'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual Learning via iCaRL.')
        parser.add_argument('--softmax_temp', type=float, default=2,
                            help='Temperature of the softmax function.')
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size)
        self.eye = torch.eye(self.num_classes).to(self.device)
        self.buffer2 = Buffer(self.args.buffer_size)

        self.class_means = None
        self.old_net = None

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        feats = self.net(x, returnt='features')
        feats = feats.view(feats.size(0), -1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.class_means = None
        if self.current_task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.old_net(inputs))
        self.opt.zero_grad()
        loss, outputs = self.get_loss(
            inputs, labels, self.current_task, logits)

        aug_repeat = self.args.aug_repeat
        prog_aug = self.args.prog_aug
        prog_alpha = self.args.prog_alpha
        distill_type = 2

        # This is View-Batch SSL implementation.
        # View-Batch Replay can be found in line 24 of the datasets/utils/continual_datasets.py.
        if aug_repeat > 1 and prog_aug in [1, 2, 3, 4] and distill_type in [1, 2, 3, 4] and epoch > 5:
            logit = F.softmax(outputs, dim=-1)
            # features = features[-1]
            # features = features.permute(0, 2, 3, 1)
            # features = features.reshape(features.size(0), -1, features.size(-1))
            # features = F.normalize(features)

            prog_loss = 0
            prog_loss_cnt = 0
            prev_label = -1
            noise_level = 0
            for b in range(logit.size(0)):
                if prev_label == labels[b] and noise_level < aug_repeat:
                    noise_level += 1
                    prog_loss_cnt += 1
                    if distill_type == 1:
                        prog_loss += F.smooth_l1_loss(features[b].unsqueeze(0), prev_feature, beta=2.)
                    if distill_type == 2:
                        prog_loss += F.kl_div(logit[b].unsqueeze(0).log(), prev_logit)
                    if distill_type == 3:
                        prog_loss += F.mse_loss(features[b].unsqueeze(0), prev_feature)
                else:
                    noise_level = 0
                    prev_label = labels[b]
                    if distill_type == 1 or distill_type == 3:
                        prev_feature = features[b].detach().clone().unsqueeze(0)
                    if distill_type == 2:
                        prev_logit = logit[b].detach().clone().unsqueeze(0)

            if prog_loss_cnt > 0:
                prog_loss /= prog_loss_cnt
                loss += prog_alpha * prog_loss

        loss.backward()

        self.opt.step()

        return loss.item()

    @staticmethod
    def binary_cross_entropy(pred, y):
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss tensor.

        Args:
            inputs: the images to be fed to the network
            labels: the ground-truth labels
            task_idx: the task index
            logits: the logits of the old network

        Returns:
            the differentiable loss value
        """

        # outputs, features = self.net(inputs, returnt='full')
        outputs = self.net(inputs, returnt='out')
        outputs = outputs[:, :self.n_seen_classes]
        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :self.n_seen_classes]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss >= 0
        else:
            targets = self.eye[labels][:,
                      self.n_past_classes:self.n_seen_classes]
            comb_targets = torch.cat(
                (logits[:, :self.n_past_classes], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0

        return loss, outputs  # , features

    def begin_task(self, dataset):
        if self.args.inference_only:
            return
        icarl_replay(self, dataset)
        self.fill_mid_task_buffer(dataset)

    def fill_mid_task_buffer(self, dataset):
        with torch.no_grad():
            fill_buffer(self.buffer2, dataset, self.current_task,
                        net=self.net, use_herding=False)

    def end_task(self, dataset) -> None:
        if self.args.inference_only:
            self.buffer2 = deepcopy(self.buffer)
            return
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        with torch.no_grad():
            fill_buffer(self.buffer, dataset, self.current_task,
                        net=self.net, use_herding=True)
        self.class_means = None
        self.buffer2 = deepcopy(self.buffer)

    @torch.no_grad()
    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        buf_data = self.buffer2.get_all_data(transform, device=self.device)
        examples, labels = buf_data[0], buf_data[1]
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                allt = None
                while len(x_buf):
                    batch = x_buf[:self.args.batch_size]
                    x_buf = x_buf[self.args.batch_size:]
                    feats = self.net(batch, returnt='features').mean(0)
                    if allt is None:
                        allt = feats
                    else:
                        allt += feats
                        allt /= 2
                class_means.append(allt.flatten())
        self.class_means = torch.stack(class_means)

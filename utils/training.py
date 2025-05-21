# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import math
import sys
from argparse import Namespace
from typing import Iterable, Tuple

import torch
from torchvision.transforms import transforms as tf
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel
from PIL import Image
from timm.data.auto_augment import rand_augment_transform

from utils import random_id
from utils.checkpoints import mammoth_load_checkpoint
from utils.loggers import *
from utils.stats import track_system_stats
from utils.status import ProgressBar
import time

try:
    import wandb
except ImportError:
    wandb = None

def reaugment(x, aug_type='autoaugment'):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    mean_tensor = torch.tensor(mean).reshape(3, 1, 1)
    std_tensor = torch.tensor(std).reshape(3, 1, 1)
    img = np.asarray(((x * std_tensor + mean_tensor) * 255).permute(1, 2, 0).to(torch.uint8))
    img = Image.fromarray(img)
    if aug_type == 'autoaugment':
        aug = tf.Compose([AutoAugment(policy=AutoAugmentPolicy.CIFAR10), tf.ToTensor(), tf.Normalize(mean, std)])
    elif aug_type == 'randaugment':
        aa_params = dict(
            translate_const=int(32 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        aug = tf.Compose([
            rand_augment_transform('rand-m7-mstd0.5-inc1', aa_params),
            tf.ToTensor(),
            tf.Normalize(mean, std)
        ])
    elif aug_type == 'randerase':
        aug = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean, std),
            tf.RandomErasing(),
        ])
    elif aug_type == 'colorjitter':
        aug = tf.Compose([
            tf.ColorJitter(0.4, 0.4, 0.4),
            tf.ToTensor(),
            tf.Normalize(mean, std)
        ])
    elif aug_type == 'randomcrop':
        aug = tf.Compose([
            tf.RandomCrop(32, padding=4),
            tf.ToTensor(),
            tf.Normalize(mean, std)
        ])
    elif aug_type == 'gaussiannoise':
        aug = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean, std),
            tf.Lambda(lambda x: x + torch.randn_like(x) * 0.1)  # Add gaussian noise with std=0.1
        ])
    elif 'gaussianblur':
        aug = tf.Compose([
            tf.GaussianBlur(kernel_size=3),
            tf.ToTensor(),
            tf.Normalize(mean, std)
        ])
    else:
        raise ValueError(f'Unknown augmentation type {aug_type}')
    x = aug(img)
    return x

def reaugment_domainnet_128(x):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    mean_tensor = torch.tensor(mean).reshape(3, 1, 1)
    std_tensor = torch.tensor(std).reshape(3, 1, 1)
    img = np.asarray(((x * std_tensor + mean_tensor) * 255).permute(1, 2, 0).to(torch.uint8))
    img = Image.fromarray(img)
    aa_params = dict(
        translate_const=int(128 * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in mean]),
    )
    aug = tf.Compose([
        rand_augment_transform('rand-m7-mstd0.5-inc1', aa_params),
        tf.ColorJitter(0.4, 0.4, 0.4),
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ])
    x = aug(img)
    return x

def reaugment_domainnet(x):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    mean_tensor = torch.tensor(mean).reshape(3, 1, 1)
    std_tensor = torch.tensor(std).reshape(3, 1, 1)
    img = np.asarray(((x * std_tensor + mean_tensor) * 255).permute(1, 2, 0).to(torch.uint8))
    img = Image.fromarray(img)
    aa_params = dict(
        translate_const=int(224 * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in mean]),
    )
    aug = tf.Compose([
        rand_augment_transform('rand-m7-mstd0.5-inc1', aa_params),
        tf.ColorJitter(0.4, 0.4, 0.4),
        tf.ToTensor(),
        tf.Normalize(mean, std)
    ])
    x = aug(img)
    return x


def reaugment_tinyimagenet(x, aug_type='randaugment'):
    mean, std =  (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    mean_tensor = torch.tensor(mean).reshape(3, 1, 1)
    std_tensor = torch.tensor(std).reshape(3, 1, 1)
    img = np.asarray(((x * std_tensor + mean_tensor) * 255).permute(1, 2, 0).to(torch.uint8))
    img = Image.fromarray(img)

    aa_params = dict(
        translate_const=int(64 * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in mean]),
    )
    if aug_type == 'autoaugment':
        aug = tf.Compose([AutoAugment(policy=AutoAugmentPolicy.IMAGENET), tf.ToTensor(), tf.Normalize(mean, std)])
    elif aug_type == 'randaugment':
        aug = tf.Compose([
            rand_augment_transform('rand-m7-mstd0.5-inc1', aa_params),
            tf.ColorJitter(0.4, 0.4, 0.4),
            tf.ToTensor(), tf.Normalize(mean, std),
        ])
    elif aug_type == 'randerase':
        aug = tf.Compose([
            tf.ColorJitter(0.4, 0.4, 0.4),
            tf.ToTensor(),
            tf.Normalize(mean, std),
            tf.RandomErasing(),
        ])
    elif aug_type == 'colorjitter':
        aug = tf.Compose([
            tf.ColorJitter(0.4, 0.4, 0.4),
            tf.ToTensor(),
            tf.Normalize(mean, std)
        ])
    elif aug_type == 'randomcrop':
        aug = tf.Compose([
            tf.RandomResizedCrop(64),
            tf.ToTensor(),
            tf.Normalize(mean, std)
        ])
    elif aug_type == 'gaussiannoise':
        aug = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean, std),
            tf.Lambda(lambda x: x + torch.randn_like(x) * 0.1)  # Add gaussian noise with std=0.1
        ])
    elif 'gaussianblur':
        aug = tf.Compose([
            tf.GaussianBlur(kernel_size=3),
            tf.ToTensor(),
            tf.Normalize(mean, std)
        ])
    else:
        raise ValueError(f'Unknown augmentation type {aug_type}')
    
    x = aug(img)
    return x


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.

    Args:
        outputs: the output tensor
        dataset: the continual dataset
        k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


@torch.no_grad()
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, return_loss=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        last: a boolean indicating whether to evaluate only the last task
        return_loss: a boolean indicating whether to return the loss in addition to the accuracy

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task. If return_loss is True, the loss is also returned as a third element.
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    loss_fn = dataset.get_loss()
    avg_loss = 0
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            if return_loss:
                loss = loss_fn(outputs, labels)
                avg_loss += loss.item()

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    if return_loss:
        return accs, accs_mask_classes, avg_loss / total
    return accs, accs_mask_classes


def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    run_name = args.wandb_name if args.wandb_name is not None else args.model

    run_id = random_id(5)
    name = f'{run_name}_{run_id}'
    wandb.init(project=args.wandb_project, config=vars(args), name=name)
    args.wandb_url = wandb.run.get_url()


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_single_epoch(model: ContinualModel,
                       train_loader: Iterable,
                       progress_bar: ProgressBar,
                       args: Namespace,
                       epoch: int,
                       current_task: int,
                       system_tracker=None,
                       data_len=None,
                       scheduler=None,
                       aug_repeat=1,
                       prog_aug=0,
                       alpha=0.2,
                       dataset="") -> int:
    """
    Trains the model for a single epoch.

    Args:
        model: the model to be trained
        train_loader: the data loader for the training set
        progress_bar: the progress bar for the current epoch
        args: the arguments from the command line
        epoch: the current epoch
        current_task: the current task index
        system_tracker: the system tracker to monitor the system stats
        data_len: the length of the training data loader. If None, the progress bar will not show the training percentage
        scheduler: the scheduler for the current epoch

    Returns:
        the number of iterations performed in the current epoch
    """
    train_iter = iter(train_loader)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    forward_time_m = AverageMeter()
    backward_time_m = AverageMeter()
    
    end = time.time()
    last_idx = len(train_loader) - 1
    num_updates = epoch * len(train_loader)

    if args.reset_bn and (epoch in [40, 60, 80] and args.dataset == 'seq-tinyimg' or epoch in [30] and args.dataset == 'seq-cifar10'):
        percent = 0.20
        model.net.reset_bn_with_percent(percent)

    i = 0
    if args.aug_type is None:
        if args.dataset == 'seq-tinyimg':
            aug_type = 'randaugment'
        elif args.dataset == 'seq-domainnet':
            aug_type = 'randaugment'
        else:
            aug_type = 'autoaugment'
    else:
        aug_type = args.aug_type

    while True:
        try:
            data = next(train_iter)
        except StopIteration:
            break
        if args.debug_mode and i > model.get_debug_iters():
            break
        if args.fitting_mode == 'iters' and progress_bar.current_task_iter >= model.args.n_iters:
            break

        if hasattr(train_loader.dataset, 'logits'):
            
            inputs, labels, not_aug_inputs, logits = data
            
            if aug_repeat == 1 and prog_aug > 0:
                for b in range(inputs.size(0)):
                    if b % 2 == 0 or b % 3 == 0:
                        if 'tinyimg' in dataset.lower():
                            inputs[b] = reaugment_tinyimagenet(inputs[b])
                        elif 'domainnet' in dataset.lower():
                            inputs[b] = reaugment_domainnet(inputs[b])
                        else:
                            inputs[b] = reaugment(inputs[b]) 

            if aug_repeat > 1 and prog_aug in [1, 2, 3, 4, 5]:
                    noise_level = 0
                    prev_label = -1
                    for b in range(inputs.size(0)):
                        if prev_label == labels[b] and noise_level < aug_repeat:
                            noise_level += 1
                            if prog_aug == 1:
                                beta = alpha * (noise_level / (aug_repeat - 1))
                                inputs[b] = inputs[b] + torch.rand_like(inputs[b]) * beta
                            elif prog_aug in [2, 5]:
                                if 'tinyimg' in dataset.lower():
                                    inputs[b] = reaugment_tinyimagenet(inputs[b], aug_type)
                                elif 'domainnet' in dataset.lower():
                                    inputs[b] = reaugment_domainnet(inputs[b])
                                else:
                                    inputs[b] = reaugment(inputs[b], aug_type)
                        else:
                            noise_level = 0
                            prev_label = labels[b]

            inputs = inputs.to(model.device)
            labels = labels.to(model.device, dtype=torch.long)
            not_aug_inputs = not_aug_inputs.to(model.device)
            logits = logits.to(model.device)

            lab_time = time.time()
            data_time_m.update(lab_time - end)
            lab_time_prev = lab_time

            loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)

            lab_time = time.time()
            forward_time_m.update(lab_time - lab_time_prev)
            lab_time_prev = lab_time
        else:
            inputs, labels, not_aug_inputs = data

            if aug_repeat == 1 and prog_aug > 0:
                for b in range(inputs.size(0)):
                    if b % 2 == 0 or b % 3 == 0:
                        if 'tinyimg' in dataset.lower():
                            inputs[b] = reaugment_tinyimagenet(inputs[b], aug_type)
                        elif 'domainnet' in dataset.lower():
                            inputs[b] = reaugment_domainnet(inputs[b], aug_type)
                        else:
                            inputs[b] = reaugment(inputs[b]) 

            if aug_repeat > 1 and prog_aug in [1, 2, 3, 4, 5]:
                    noise_level = 0
                    prev_label = -1
                    for b in range(inputs.size(0)):
                        if prev_label == labels[b] and noise_level < aug_repeat:
                            noise_level += 1
                            if prog_aug == 1:
                                beta = alpha * (noise_level / (aug_repeat - 1))
                                inputs[b] = inputs[b] + torch.rand_like(inputs[b]) * beta
                            elif prog_aug in [2, 5]:
                                if 'tinyimg' in dataset.lower():
                                    inputs[b] = reaugment_tinyimagenet(inputs[b])
                                elif 'domainnet' in dataset.lower():
                                    inputs[b] = reaugment_domainnet(inputs[b])
                                else:
                                    inputs[b] = reaugment(inputs[b])
                        else:
                            noise_level = 0
                            prev_label = labels[b]

            inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
            not_aug_inputs = not_aug_inputs.to(model.device)

            lab_time = time.time()
            data_time_m.update(lab_time - end)
            lab_time_prev = lab_time

            loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)

            lab_time = time.time()
            forward_time_m.update(lab_time - lab_time_prev)
            lab_time_prev = lab_time
        assert not math.isnan(loss)

        if args.code_optimization == 0:
            torch.cuda.synchronize()
        progress_bar.prog(i, data_len, epoch, current_task, loss)
        system_tracker()
        i += 1

        lab_time = time.time()
        # backward_time_m.update(lab_time - lab_time_prev)
        batch_time_m.update(lab_time - end)

        print(
            'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
            'Time: {batch_time.avg:.3f}s, {rate_avg:>7.2f}/s  '
            'D: {data_time.avg:.3f}  '
            'F+B+O: {forward_time.avg:.3f}  '.format(
                epoch,
                i, len(train_loader),
                100. * i / last_idx,
                # loss=losses_m,
                batch_time=batch_time_m,
                forward_time=forward_time_m,
                # backward_time=backward_time_m,
                # rate=inputs.size(0) * 1 / batch_time_m.val,
                rate_avg=inputs.size(0) * 1 / batch_time_m.avg,
                # lr=lr,
                data_time=data_time_m
            )
        )
        end = time.time()

    if scheduler is not None:
        scheduler.step()

    return i


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        initialize_wandb(args)

    if not args.disable_log:
        logger = Logger(args, dataset.SETTING, dataset.NAME, model.NAME)

    model.net.to(model.device)
    torch.cuda.empty_cache()

    with track_system_stats(logger) as system_tracker:
        results, results_mask_classes = [], []

        if args.start_from is not None:
            for i in range(args.start_from):
                train_loader, _ = dataset.get_data_loaders()
                model.meta_begin_task(dataset)
                model.meta_end_task(dataset)

        if args.loadcheck is not None:
            model, past_res = mammoth_load_checkpoint(args, model)

            if not args.disable_log and past_res is not None:
                (results, results_mask_classes, csvdump) = past_res
                logger.load(csvdump)

            print('Checkpoint Loaded!')

        progress_bar = ProgressBar(joint=args.joint, verbose=not args.non_verbose)

        if args.enable_other_metrics:
            dataset_copy = get_dataset(args)
            for t in range(dataset.N_TASKS):
                model.net.train()
                _, _ = dataset_copy.get_data_loaders()
            if model.NAME != 'icarl' and model.NAME != 'pnn':
                random_results_class, random_results_task = evaluate(model, dataset_copy)

        print(file=sys.stderr)
        start_task = 0 if args.start_from is None else args.start_from
        end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

        torch.cuda.empty_cache()
        print(f"Starting training from task {start_task} to task {end_task}")
        for t in range(start_task, end_task):
            model.net.train()
            train_loader, test_loader = dataset.get_data_loaders()
            model.meta_begin_task(dataset)
            # HACK: to update the dataset_len in the sampler, becuase iCARL merges the replay buffer with the current task dataset after creating the samplers
            if hasattr(train_loader.sampler, 'set_dataset_len'):
                train_loader.sampler.set_dataset_len(len(train_loader.dataset))
            if not args.inference_only:
                if t and args.enable_other_metrics:
                    accs = evaluate(model, dataset, last=True)
                    results[t - 1] = results[t - 1] + accs[0]
                    if dataset.SETTING == 'class-il':
                        results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

                scheduler = dataset.get_scheduler(model, args, reload_optim=True) if not hasattr(model, 'scheduler') else model.scheduler

                epoch = 0
                best_ea_metric = None
                best_ea_model = None
                cur_stopping_patience = args.early_stopping_patience
                while True:
                    data_len = None
                    if not isinstance(dataset, GCLDataset):
                        data_len = len(train_loader)

                    if hasattr(train_loader.sampler, 'set_epoch'):
                        train_loader.sampler.set_epoch(epoch)

                    train_single_epoch(model, train_loader, progress_bar, args, current_task=t, epoch=epoch,
                                       system_tracker=system_tracker, data_len=data_len, scheduler=scheduler, 
                                       aug_repeat=args.aug_repeat, prog_aug=args.prog_aug, dataset=dataset.NAME)

                    epoch += 1
                    if args.fitting_mode == 'epochs' and epoch >= model.args.n_epochs:
                        break
                    elif args.fitting_mode == 'iters' and progress_bar.current_task_iter >= model.args.n_iters:
                        break
                    elif args.fitting_mode == 'early_stopping' and epoch % args.early_stopping_freq == 0 and epoch > 0:
                        epoch_accs, _, epoch_loss = evaluate(model, dataset, return_loss=True, last=True)

                        if args.early_stopping_metric == 'accuracy':
                            ea_metric = np.mean(epoch_accs)  # Higher accuracy is better
                        elif args.early_stopping_metric == 'loss':
                            ea_metric = -epoch_loss  # Lower loss is better
                        else:
                            raise ValueError(f'Unknown early stopping metric {args.early_stopping_metric}')

                        # Higher accuracy is better
                        if best_ea_metric is not None and ea_metric - best_ea_metric < args.early_stopping_epsilon:
                            cur_stopping_patience -= args.early_stopping_freq
                            if cur_stopping_patience <= 0:
                                print(f"\nEarly stopping at epoch {epoch} with metric {abs(ea_metric)}", file=sys.stderr)
                                model.load_state_dict({k: v.to(model.device) for k, v in best_ea_model.items()})
                                break
                            print(f"\nNo improvement at epoch {epoch} (best {abs(best_ea_metric)} | current {abs(ea_metric)}). "
                                  f"Waiting for {cur_stopping_patience} epochs to stop.", file=sys.stderr)
                        else:
                            print(f"\nFound better model with metric {abs(ea_metric)} at epoch {epoch}. "
                                  f"Previous value was {abs(best_ea_metric) if best_ea_metric is not None else 'None'}", file=sys.stderr)
                            best_ea_metric = ea_metric
                            best_ea_model = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                            cur_stopping_patience = args.early_stopping_patience

                    if args.eval_epochs is not None and (epoch > 0 or args.eval_epochs == 1) and epoch % args.eval_epochs == 0 and epoch < model.args.n_epochs:
                        epoch_accs = evaluate(model, dataset)

                        log_accs(args, logger, epoch_accs, t, dataset.SETTING, epoch=epoch)

            progress_bar.reset()

            model.meta_end_task(dataset)

            accs = evaluate(model, dataset)
            results.append(accs[0])
            results_mask_classes.append(accs[1])


            log_accs(args, logger, accs, t, dataset.SETTING)

            if args.savecheck:
                save_obj = {
                    'model': model.state_dict(),
                    'args': args,
                    'results': [results, results_mask_classes, logger.dump()],
                    'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                }
                if 'buffer_size' in model.args:
                    save_obj['buffer'] = deepcopy(model.buffer).to('cpu')

                # Saving model checkpoint
                checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_{t}.pt'
                torch.save(save_obj, checkpoint_name)

        del progress_bar

        if args.validation:
            # Final evaluation on the real test set
            print("Starting final evaluation on the real test set...", file=sys.stderr)
            del dataset
            args.validation = None
            args.validation_mode = 'current'

            final_dataset = get_dataset(args)
            for _ in range(final_dataset.N_TASKS):
                final_dataset.get_data_loaders()
            accs = evaluate(model, final_dataset)

            log_accs(args, logger, accs, 'final', final_dataset.SETTING, prefix="FINAL")

        if not args.disable_log and args.enable_other_metrics:
            logger.add_bwt(results, results_mask_classes)
            logger.add_forgetting(results, results_mask_classes)
            if model.NAME != 'icarl' and model.NAME != 'pnn':
                logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

        system_tracker.print_stats()

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()

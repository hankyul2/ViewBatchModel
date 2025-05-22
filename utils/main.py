"""
This script is the main entry point for the Mammoth project. It contains the main function `main()` that orchestrates the training process.

The script performs the following tasks:
- Imports necessary modules and libraries.
- Sets up the necessary paths and configurations.
- Parses command-line arguments.
- Initializes the dataset, model, and other components.
- Trains the model using the `train()` function.

To run the script, execute it directly or import it as a module and call the `main()` function.
"""
# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# needed (don't change it)
import pprint
import numpy  # noqa
import time
import importlib
import os
import socket
import sys
import datetime
import uuid
from argparse import ArgumentParser
import torch

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')
# sys.path.remove('/home/hankyul/private/LightNet/edede')  # for debugging purpose only

from utils import create_if_not_exists, custom_str_underscore
from utils.args import add_management_args, add_experiment_args
from utils.conf import base_path, get_device
from utils.distributed import make_dp
from utils.best_args import best_args
from utils.conf import set_random_seed


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    """
    Parse command line arguments for the mammoth program and sets up the `args` object.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    from models import get_all_models, get_model_class
    from datasets import get_dataset_names, get_dataset
    # from datasets.utils import update_default_args

    parser = ArgumentParser(description='mammoth', allow_abbrev=False, add_help=False)
    parser.add_argument('--aug-repeat', type=int, default=1)
    parser.add_argument('--prog-aug', type=int, default=0)
    parser.add_argument('--distill-type', type=int, default=2, help='Distillation type.')
    parser.add_argument('--flag', type=str, default=None, help='Flag for the experiment.')
    parser.add_argument('--no_buffer', action='store_true', help='No buffer.')
    parser.add_argument('--only_buffer', action='store_true', help='Only buffer.')
    parser.add_argument('--prog-alpha', type=float, default=1.0, help='Progressive alpha.')
    parser.add_argument('--model', type=custom_str_underscore, help='Model name.',
                        choices=list(get_all_models().keys()))
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    parser.add_argument('--joint', type=int, choices=[0, 1], default=0, help='Joint training.')
    parser.add_argument('--debug_mode', type=int, choices=[0, 1], default=0, help='Debug mode.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_epochs', type=int, default=None, help='Number of epochs to evaluate the model.')
    parser.add_argument('--reset-bn', action='store_true', help='Reset batch norm statistics.')
    parser.add_argument('--freq-filter', type=str, default=None, help='Frequency filter type.')
    parser.add_argument('--permute_classes', type=int, default=0, help='Permute classes.')
    parser.add_argument('--distributed', type=str, default=None, choices=[None, 'dp', 'ddp'],
                        help='Distributed training.')
    parser.add_argument('--imb_ratio', type=float, default=1.0, help='Imbalance ratio.')
    parser.add_argument('--aug-type', type=str, default=None, help='Augmentation type.')
    parser.add_argument('--inference_only', default=0, choices=[0, 1], type=int,
                        help='Perform inference only for each task (no training).')
    parser.add_argument('--loadcheck', type=str, default=None,
                        help='Path of the checkpoint to load (.pt file for the specific task)')
    parser.add_argument('--start_from', type=int, default=None, help="Task to start from")
    parser.add_argument('--stop_after', type=int, default=None, help="Task limit")
    parser.add_argument('--savecheck', default=0, choices=[0, 1], type=int, help='Save checkpoint?')
    parser.add_argument('--ckpt_name', type=str, required=False, help='Checkpoint name.')

    args = parser.parse_known_args()[0]
    models_dict = get_all_models()
    if args.model is None:
        print('No model specified. Please specify a model with --model to see all other options.')
        print('Available models are: {}'.format(list(models_dict.keys())))
        sys.exit(1)

    mod = importlib.import_module('models.' + models_dict[args.model])

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=get_dataset_names(),
                            help='Which dataset to perform experiments on.')
        parser.add_argument('--seed', type=int, default=1993)
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]

        parser = get_model_class(args).get_parser()
        parser.add_argument('--aug-repeat', type=int, default=1)
        parser.add_argument('--prog-aug', type=int, default=0)
        parser.add_argument('--distill-type', type=int, default=2, help='Distillation type.')
        parser.add_argument('--flag', type=str, default=None, help='Flag for the experiment.')
        parser.add_argument('--no_buffer', action='store_true', help='No buffer.')
        parser.add_argument('--only_buffer', action='store_true', help='Only buffer.')
        parser.add_argument('--prog-alpha', type=float, default=1.0, help='Progressive alpha.')
        parser.add_argument('--reset-bn', action='store_true', help='Reset batch norm statistics.')
        parser.add_argument('--freq-filter', type=str, default=None, help='Frequency filter type.')
        parser.add_argument('--imb_ratio', type=float, default=1.0, help='Imbalance ratio.')
        parser.add_argument('--aug-type', type=str, default=None, help='Augmentation type.')

        add_management_args(parser)
        add_experiment_args(parser)
        lr = best['lr']
        tmp = []
        if 'alpha' in best:
            tmp = ['--alpha', str(best['alpha'])]
        if 'beta' in best:
            tmp += ['--beta', str(best['beta'])]
        to_parse = sys.argv[1:] + ['--lr', str(lr)] + tmp
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)

        for key, value in best.items():
            setattr(args, key, value)

        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        parser = get_model_class(args).get_parser()
        parser.add_argument('--aug-repeat', type=int, default=1)
        parser.add_argument('--prog-aug', type=int, default=0)
        parser.add_argument('--distill-type', type=int, default=2, help='Distillation type.')
        parser.add_argument('--flag', type=str, default=None, help='Flag for the experiment.')
        parser.add_argument('--no_buffer', action='store_true', help='No buffer.')
        parser.add_argument('--reset-bn', action='store_true', help='Reset batch norm statistics.')
        parser.add_argument('--freq-filter', type=str, default=None, help='Frequency filter type.')
        parser.add_argument('--imb_ratio', type=float, default=1.0, help='Imbalance ratio.')
        parser.add_argument('--aug-type', type=str, default=None, help='Augmentation type.')
        add_management_args(parser)
        add_experiment_args(parser)
        args = parser.parse_args()

    get_dataset(args).update_default_args()
    args.model = models_dict[args.model]

    if args.lr_scheduler is not None:
        print('Warning: lr_scheduler set to {}, overrides default from dataset.'.format(args.lr_scheduler),
              file=sys.stderr)

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.savecheck:
        assert args.inference_only == 0, "Should not save checkpoint in inference only mode"
        if not os.path.isdir('checkpoints'):
            create_if_not_exists("checkpoints")

        now = time.strftime("%Y%m%d-%H%M%S")
        extra_ckpt_name = "" if args.ckpt_name is None else f"{args.ckpt_name}_"
        args.ckpt_name = f"{extra_ckpt_name}{args.model}_{args.dataset}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.n_epochs}_{str(now)}"
        args.ckpt_name_replace = f"{extra_ckpt_name}{args.model}_{args.dataset}_{'{}'}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}__{args.n_epochs}_{str(now)}"
        print("Saving checkpoint into", args.ckpt_name, file=sys.stderr)

    if args.joint:
        assert args.start_from is None and args.stop_after is None, "Joint training does not support start_from and stop_after"
        assert args.enable_other_metrics == 0, "Joint training does not support other metrics"

    assert 0 < args.label_perc <= 1, "label_perc must be in (0, 1]"

    if args.validation is not None:
        print(f"INFO: Using {args.validation}% of the training set as validation set.", file=sys.stderr)
        print(f"INFO: Validation will be computed with mode `{args.validation_mode}`.", file=sys.stderr)

    return args


def main(args=None):
    from models import get_model
    from datasets import ContinualDataset, get_dataset
    from utils.training import train

    # torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_num_threads(8)

    lecun_fix()
    if args is None:
        args = parse_args()

    device = get_device()
    args.device = device

    # set base path
    base_path(args.base_path)

    if args.code_optimization != 0:
        torch.set_float32_matmul_precision('high' if args.code_optimization == 1 else 'medium')
        print("INFO: code_optimization is set to", args.code_optimization, file=sys.stderr)
        print(f"Using {torch.get_float32_matmul_precision()} precision for matmul.", file=sys.stderr)

        if args.code_optimization == 2:
            if not torch.cuda.is_bf16_supported():
                raise NotImplementedError('BF16 is not supported on this machine.')

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    if args.fitting_mode == 'epochs' and args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    elif args.fitting_mode == 'iters' and args.n_iters is None and isinstance(dataset, ContinualDataset):
        args.n_iters = dataset.get_iters()

    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
        if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and (
                not hasattr(args, 'minibatch_size') or args.minibatch_size is None):
            args.minibatch_size = dataset.get_minibatch_size()
    else:
        args.minibatch_size = args.batch_size

    if args.validation:
        if args.validation_mode == 'current':
            assert dataset.SETTING in ['class-il',
                                       'task-il'], "`current` validation modes is only supported for class-il and task-il settings (requires a task division)."

    backbone = dataset.get_backbone()
    if args.code_optimization == 3:
        # check if the model is compatible with torch.compile
        # from https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
        if torch.cuda.get_device_capability()[0] >= 7 and os.name != 'nt':
            print("================ Compiling model with torch.compile ================")
            print("WARNING: `torch.compile` may break your code if you change the model after the first run!")
            print("This includes adding classifiers for new tasks, changing the backbone, etc.")
            print(
                "ALSO: some models CHANGE the backbone during initialization. Remember to call `torch.compile` again after that.")
            print("====================================================================")
            backbone = torch.compile(backbone)
        else:
            if torch.cuda.get_device_capability()[0] < 7:
                raise NotImplementedError('torch.compile is not supported on this machine.')
            else:
                raise Exception(
                    f"torch.compile is not supported on Windows. Check https://github.com/pytorch/pytorch/issues/90768 for updates.")

    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    # model = torch.compile(model)

    if args.distributed == 'dp':
        if args.batch_size < torch.cuda.device_count():
            raise Exception(f"Batch too small for DataParallel (Need at least {torch.cuda.device_count()}).")

        model.net = make_dp(model.net)
        model.to('cuda:0')
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError('Distributed Data Parallel not supported yet.')

    if args.debug_mode:
        print('Debug mode enabled: running only a few forward steps per epoch with W&B disabled.')
        args.nowand = 1

    if args.wandb_entity is None or args.wandb_project is None:
        print('Warning: wandb_entity and wandb_project not set. Disabling wandb.')
        args.nowand = 1
    else:
        print('Logging to wandb: {}/{}'.format(args.wandb_entity, args.wandb_project))
        args.nowand = 0

    try:
        import setproctitle
        # set job name
        setproctitle.setproctitle(
            '{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    except Exception:
        pass

    train(model, dataset, args)


if __name__ == '__main__':
    main()

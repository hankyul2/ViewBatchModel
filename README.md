# Do Your Best and Get Enough Rest for Continual Learning

This folder contains official pyTorch implementations for *"Do Your Best and Get Enough Rest for Continual Learning"* accepted in CVPR'25. (see our [paper](https://arxiv.org/pdf/2503.18371), [slides](https://cvpr.thecvf.com/media/cvpr-2025/Slides/34881.pdf), [poster](https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202025/34881.png?t=1747793527.8730633)).



<p align="center">
    <img width="650px" src="https://github.com/user-attachments/assets/2eb0685f-d4eb-4d06-a8a1-341c62641c53"/>
    <br/>
  <h4 align="center">Illustration of View-Batch Model</h4>
</p>



## 1. Tutorial

1. Clone this repository and install the requirements.

   ```bash
   git clone https://github.com/hankyul2/ViewBatchModel.git
   cd ViewBatchModel
   pip install -r requirements.txt
   ```

   

2. Train ResNet18 on S-CIFAR-10 using iCaRL as baseline methods with 200 buffers.

   iCaRL

   ```bash
   # this will save checkpoints to icarl_r1_s1993
   CUDA_VISIBLE_DEVICES=0 python utils/main.py --model icarl --load_best_args --dataset seq-cifar10 --buffer_size 200 --seed 1993 --savecheck 1 --ckpt_name icarl_r1_s1993
   ```

   **Ours**-iCaRL

   ```bash
   CUDA_VISIBLE_DEVICES=4 python utils/main.py --model icarl --load_best_args --dataset seq-cifar10 --buffer_size 200 --aug-repeat 4 --prog-aug 5 --seed 1993 --flag hard_aug --savecheck 1 --ckpt_name icarl_r4_hard_aug_s1993
   ```

3. Validate the trained network using the saved checkpoint.

   iCaRL

   ```bash
   CUDA_VISIBLE_DEVICES=0 python utils/main.py --model icarl --load_best_args --dataset seq-cifar10 --buffer_size 200 --seed 1993 --loadcheck checkpoints/icarl_r1_s1993_cifar10_t0.pth --start_from 0 --stop_after 0 --inference_only 1
   ```

   **Ours**-iCaRL

   ```bash
   CUDA_VISIBLE_DEVICES=4 python utils/main.py --model icarl --load_best_args --dataset seq-cifar10 --buffer_size 200 --seed 1997 --loadcheck checkpoints/icarl_r4_hard_aug_s1997_cifar10_t0.pth --start_from 0 --stop_after 0 --inference_only 1
   ```

4. See [scripts/icarl](scripts/icarl) for more commands to reproduce Table 6 in the paper. Also, check [datasets/utils/continual_dataset.py#L24](datasets/utils/continual_dataset.py#L24) for view-batch replay and [models/icarl.py#L78](models/icarl.py#L78) for view-batch SSL.



## 2. Reproduced Results

After the paper has been accepted, we rerun everything to provide complete logs and checkpoints for our Table 6 in the paper. Our exact environments are:

- `torch==1.12.1+cu113`
- `torchvision==0.13.1+cu113`
- `timm==1.0.7`
- `numpy==1.24.4`



### Experimental Results

The table below reproduces Table 6 of our paper, which contains the main ablation study for the proposed method.

| Method | View-batch Replay | Strong Augment | View-batch SSL | Forgetting(⬇️) | CIL(⬆️)    | TIL(⬆️)    | AVG   | ∆         |
|--------|-------------------|----------------|----------------|----------------|------------|------------|-------|-----------|
| iCaRL  | ❌                 | ❌              | ❌              | 28.05±4.21     | 63.58±2.64 | 90.32±3.19 | 76.95 | -         |
| iCaRL  | ❌                 | ✅              | ❌              | 22.16±0.91     | 65.33±1.05 | 89.33±0.58 | 77.33 | **+0.38** |
| iCaRL  | ✅                 | ❌              | ❌              | 18.72±1.76     | 67.21±0.42 | 91.63±0.98 | 79.42 | **+2.47** |
| iCaRL  | ✅                 | ✅              | ❌              | 18.29±0.91     | 67.16±0.75 | 91.02±0.97 | 79.09 | **+2.14** |
| iCaRL  | ✅                 | ✅              | ✅              | 13.81±1.58     | 69.25±0.41 | 92.73±0.57 | 80.99 | **+4.04** |



### Log

Below, the WanDB project link provides the complete logs that are made during the training of the above tables. It includes:

- command line
- metrics
- console outputs
- environments

**WanDB Project Link**: https://wandb.ai/gregor99/view_batch_model.



### Checkpoint

The tables below provide the checkpoints saved at the end of tasks during the training of the above tables.

<details>
    <summary>seed=1993</summary>

| method | View-batch Replay | Strong Augmentation | View-batch SSL | task 1                                                                                                            | task 2                                                                                                            | task 3                                                                                                            | task 4                                                                                                            | task 5                                                                                                            |
|--------|-------------------|---------------------|----------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| iCaRL  | -                 | -                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1993_cifar10_t0.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1993_cifar10_t1.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1993_cifar10_t2.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1993_cifar10_t3.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1993_cifar10_t4.pt)          |
| iCaRL  | -                 | v                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1993_cifar10_t0.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1993_cifar10_t1.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1993_cifar10_t2.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1993_cifar10_t3.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1993_cifar10_t4.pt) |
| iCaRL  | v                 | -                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1993_cifar10_t0.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1993_cifar10_t1.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1993_cifar10_t2.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1993_cifar10_t3.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1993_cifar10_t4.pt)          |
| iCaRL  | v                 | v                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1993_cifar10_t0.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1993_cifar10_t1.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1993_cifar10_t2.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1993_cifar10_t3.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1993_cifar10_t4.pt) |
| iCaRL  | v                 | v                   | v              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1993_cifar10_t0.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1993_cifar10_t1.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1993_cifar10_t2.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1993_cifar10_t3.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1993_cifar10_t4.pt)      |

</details>



<details>
    <summary>seed=1996</summary>

| method | View-batch Replay | Strong Augmentation | View-batch SSL | task 1                                                                                                            | task 2                                                                                                            | task 3                                                                                                            | task 4                                                                                                            | task 5                                                                                                            |
|--------|-------------------|---------------------|----------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| iCaRL  | -                 | -                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1996_cifar10_t0.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1996_cifar10_t1.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1996_cifar10_t2.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1996_cifar10_t3.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1996_cifar10_t4.pt)          |
| iCaRL  | -                 | v                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1996_cifar10_t0.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1996_cifar10_t1.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1996_cifar10_t2.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1996_cifar10_t3.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1996_cifar10_t4.pt) |
| iCaRL  | v                 | -                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1996_cifar10_t0.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1996_cifar10_t1.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1996_cifar10_t2.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1996_cifar10_t3.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1996_cifar10_t4.pt)          |
| iCaRL  | v                 | v                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1996_cifar10_t0.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1996_cifar10_t1.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1996_cifar10_t2.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1996_cifar10_t3.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1996_cifar10_t4.pt) |
| iCaRL  | v                 | v                   | v              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1996_cifar10_t0.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1996_cifar10_t1.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1996_cifar10_t2.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1996_cifar10_t3.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1996_cifar10_t4.pt)      |

</details>



<details>
    <summary>seed=1997</summary>

| method | View-batch Replay | Strong Augmentation | View-batch SSL | task 1                                                                                                            | task 2                                                                                                            | task 3                                                                                                            | task 4                                                                                                            | task 5                                                                                                            |
|--------|-------------------|---------------------|----------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| iCaRL  | -                 | -                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1997_cifar10_t0.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1997_cifar10_t1.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1997_cifar10_t2.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1997_cifar10_t3.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_s1997_cifar10_t4.pt)          |
| iCaRL  | -                 | v                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1997_cifar10_t0.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1997_cifar10_t1.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1997_cifar10_t2.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1997_cifar10_t3.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r1_hard_aug_s1997_cifar10_t4.pt) |
| iCaRL  | v                 | -                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1997_cifar10_t0.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1997_cifar10_t1.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1997_cifar10_t2.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1997_cifar10_t3.pt)          | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_s1997_cifar10_t4.pt)          |
| iCaRL  | v                 | v                   | -              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1997_cifar10_t0.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1997_cifar10_t1.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1997_cifar10_t2.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1997_cifar10_t3.pt) | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_hard_aug_s1997_cifar10_t4.pt) |
| iCaRL  | v                 | v                   | v              | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1997_cifar10_t0.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1997_cifar10_t1.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1997_cifar10_t2.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1997_cifar10_t3.pt)      | [ckpt](https://github.com/hankyul2/ViewBatchModel/releases/download/v1.0.0/icarl_r4_ssl_s1997_cifar10_t4.pt)      |

</details>



## 3. Acknowledgement

This project is heavily based on [Mammoth](https://github.com/aimagelab/mammoth). We sincerely appreciate the authors of the mentioned works for sharing such great library as open-source project.

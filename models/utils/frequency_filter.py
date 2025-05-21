import numpy as np
import cv2
import torch
import torchvision.transforms as tf
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from timm.data.auto_augment import rand_augment_transform
from numba import jit, cuda
import cupy as cp  # Optional: for GPU support


def high_pass_filter(images, cutoff):
    # Convert to float32
    images = images.astype(np.float32)
    
    # Create filter mask once
    rows, cols = images.shape[1:3]
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
    mask = (x*x + y*y >= cutoff*cutoff).astype(np.float32)
    
    # Process entire batch
    if len(images.shape) == 4:  # RGB
        fft = np.fft.fft2(images, axes=(1, 2))
        fft_shift = np.fft.fftshift(fft, axes=(1, 2))
        fft_shift_filtered = fft_shift * mask[None, :, :, None]
        filtered_batch = np.fft.ifft2(np.fft.ifftshift(fft_shift_filtered, axes=(1, 2)), axes=(1, 2))
        filtered_batch = np.abs(filtered_batch).astype(np.uint8)
    else:  # Grayscale
        fft = np.fft.fft2(images, axes=(1, 2))
        fft_shift = np.fft.fftshift(fft, axes=(1, 2))
        fft_shift_filtered = fft_shift * mask[None, :, :]
        filtered_batch = np.fft.ifft2(np.fft.ifftshift(fft_shift_filtered, axes=(1, 2)), axes=(1, 2))
        filtered_batch = np.abs(filtered_batch).astype(np.uint8)
    
    return filtered_batch


def low_pass_filter(image, cutoff, use_gpu=False):
    if use_gpu and cuda.is_available():
        return _low_pass_filter_gpu(image, cutoff)
    return _low_pass_filter_cpu(image, cutoff)

# @jit(nopython=True)
def _low_pass_filter_cpu(image, cutoff):
    # Convert to float32 for better performance
    image = image.astype(np.float32)
    
    # Create filter mask once for all channels
    rows, cols = image.shape[:2]
    center_row, center_col = rows // 2, cols // 2
    y = np.arange(-center_row, rows-center_row)[:, np.newaxis]
    x = np.arange(-center_col, cols-center_col)[np.newaxis, :]
    mask = ((x*x + y*y) <= cutoff*cutoff).astype(np.float32)
    
    # Process all channels at once
    if len(image.shape) == 3:
        # FFT
        fft = np.fft.fft2(image, axes=(0, 1))
        fft_shift = np.fft.fftshift(fft)
        
        # Apply mask to all channels simultaneously
        fft_shift_filtered = fft_shift * mask[:, :, np.newaxis]
        
        # Inverse FFT
        filtered_image = np.fft.ifft2(np.fft.ifftshift(fft_shift_filtered))
        filtered_image = np.abs(filtered_image).astype(np.uint8)
    else:
        # Handle grayscale images
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        fft_shift_filtered = fft_shift * mask
        filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shift_filtered))).astype(np.uint8)
    
    return filtered_image

# @jit(nopython=True)
def _low_pass_filter_gpu(image, cutoff):
    with cp.cuda.Device(0):
        # Move image to GPU
        image_gpu = cp.asarray(image, dtype=cp.float32)
        
        # Create filter mask on GPU
        rows, cols = image.shape[:2]
        center_row, center_col = rows // 2, cols // 2
        y, x = cp.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
        mask = (x*x + y*y <= cutoff*cutoff).astype(cp.float32)
        
        if len(image.shape) == 3:
            fft = cp.fft.fft2(image_gpu, axes=(0, 1))
            fft_shift = cp.fft.fftshift(fft)
            fft_shift_filtered = fft_shift * mask[:, :, cp.newaxis]
            filtered_image = cp.abs(cp.fft.ifft2(cp.fft.ifftshift(fft_shift_filtered)))
        else:
            fft = cp.fft.fft2(image_gpu)
            fft_shift = cp.fft.fftshift(fft)
            fft_shift_filtered = fft_shift * mask
            filtered_image = cp.abs(cp.fft.ifft2(cp.fft.ifftshift(fft_shift_filtered)))
        
        return cp.asnumpy(filtered_image).astype(np.uint8)

def low_pass_filter_batch(images, cutoff, use_gpu=False):
    """
    Apply low pass filter to batch of images
    Args:
        images: numpy array of shape (B,H,W,C) or (B,H,W) for grayscale
        cutoff: filter cutoff frequency
        use_gpu: whether to use GPU acceleration
    Returns:
        filtered batch of images
    """
    if use_gpu and cuda.is_available():
        return _low_pass_filter_batch_gpu(images, cutoff)
    return _low_pass_filter_batch_cpu(images, cutoff)

# @jit(nopython=True)
def _low_pass_filter_batch_cpu(images, cutoff):
    # Convert to float32
    images = images.astype(np.float32)
    
    # Create filter mask once
    rows, cols = images.shape[1:3]
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
    mask = (x*x + y*y <= cutoff*cutoff).astype(np.float32)
    
    # Process entire batch
    if len(images.shape) == 4:  # RGB
        fft = np.fft.fft2(images, axes=(1, 2))
        fft_shift = np.fft.fftshift(fft, axes=(1, 2))
        fft_shift_filtered = fft_shift * mask[None, :, :, None]
        filtered_batch = np.fft.ifft2(np.fft.ifftshift(fft_shift_filtered, axes=(1, 2)), axes=(1, 2))
        filtered_batch = np.abs(filtered_batch).astype(np.uint8)
    else:  # Grayscale
        fft = np.fft.fft2(images, axes=(1, 2))
        fft_shift = np.fft.fftshift(fft, axes=(1, 2))
        fft_shift_filtered = fft_shift * mask[None, :, :]
        filtered_batch = np.fft.ifft2(np.fft.ifftshift(fft_shift_filtered, axes=(1, 2)), axes=(1, 2))
        filtered_batch = np.abs(filtered_batch).astype(np.uint8)
    
    return filtered_batch

def _low_pass_filter_batch_gpu(images, cutoff):
    with cp.cuda.Device(0):
        # Move batch to GPU
        images_gpu = cp.asarray(images, dtype=cp.float32)
        
        # Create filter mask
        rows, cols = images.shape[1:3]
        center_row, center_col = rows // 2, cols // 2
        y, x = cp.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
        mask = (x*x + y*y <= cutoff*cutoff).astype(cp.float32)
        
        if len(images.shape) == 4:  # RGB
            fft = cp.fft.fft2(images_gpu, axes=(1, 2))
            fft_shift = cp.fft.fftshift(fft, axes=(1, 2))
            fft_shift_filtered = fft_shift * mask[None, :, :, None]
            filtered_batch = cp.abs(cp.fft.ifft2(cp.fft.ifftshift(fft_shift_filtered, axes=(1, 2)), axes=(1, 2)))
        else:  # Grayscale
            fft = cp.fft.fft2(images_gpu, axes=(1, 2))
            fft_shift = cp.fft.fftshift(fft, axes=(1, 2))
            fft_shift_filtered = fft_shift * mask[None, :, :]
            filtered_batch = cp.abs(cp.fft.ifft2(cp.fft.ifftshift(fft_shift_filtered, axes=(1, 2)), axes=(1, 2)))
        
        return torch.as_tensor(filtered_batch, dtype=torch.uint8, device='cuda')

def frequency_filter(x, filter, cutoff=10, device='cuda'):
    mean, std =  (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    mean_tensor = torch.tensor(mean, device=device).reshape(3, 1, 1)
    std_tensor = torch.tensor(std, device=device).reshape(3, 1, 1)
    
    # Handle batch dimension
    if len(x.shape) == 4:  # (B, C, H, W)
        # Expand mean and std for batch dimension
        mean_tensor = mean_tensor.unsqueeze(0)  # (1, 3, 1, 1)
        std_tensor = std_tensor.unsqueeze(0)   # (1, 3, 1, 1)
        
        # Convert batch to numpy array
        img_batch = ((x * std_tensor + mean_tensor) * 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        
        if filter == 'high':
            # Process each image in batch
            filtered_batch = high_pass_filter(img_batch, cutoff)
        elif filter == 'low':
            filtered_batch = low_pass_filter_batch(img_batch, cutoff, use_gpu=False)
        else:
            raise ValueError(f'Invalid filter type: {filter}')
        
        # Convert back to tensor
        filtered_batch = torch.stack([F.to_tensor(img) for img in filtered_batch]).to(device)
        filtered_batch = (filtered_batch - mean_tensor) / std_tensor
        return filtered_batch
    
    else:  # Single image (C, H, W)
        img = np.asarray(((x * std_tensor + mean_tensor) * 255).permute(1, 2, 0).to(torch.uint8))
        
        if filter == 'high':
            filtered_img = high_pass_filter(img[np.newaxis, ...], cutoff)[0]
        elif filter == 'low':
            filtered_img = low_pass_filter(img, cutoff, use_gpu=False)
        else:
            raise ValueError(f'Invalid filter type: {filter}')

        filtered_img = F.to_tensor(filtered_img).to(device)
        filtered_img = (filtered_img - mean_tensor) / std_tensor
        return filtered_img

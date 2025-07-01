''' SpecAugment and Gaussian Noise for Spectrograms '''

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as F

class SpecAugment(nn.Module):
    """
    SpecAugment data augmentation for spectrograms.
    
    Applies time masking (vertical bands) and frequency masking (horizontal bands)
    to spectrograms during training only.
    
    Args:
        time_mask_param (int): Maximum number of consecutive time steps to mask
        freq_mask_param (int): Maximum number of consecutive frequency bins to mask
        num_time_masks (int): Number of time masks to apply
        num_freq_masks (int): Number of frequency masks to apply
        mask_value (float): Value to use for masked regions (0.0 = black, 1.0 = white)
        p (float): Probability of applying augmentation
    """
    
    def __init__(self, time_mask_param=40, freq_mask_param=15, 
                 num_time_masks=1, num_freq_masks=1, 
                 mask_value=0.0, p=0.8):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.mask_value = mask_value
        self.p = p
    
    def forward(self, spec):
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spec (torch.Tensor): Input spectrogram of shape (C, H, W) or (H, W)
                                For spectrograms: (1, 224, 313) - (channels, freq_bins, time_steps)
        
        Returns:
            torch.Tensor: Augmented spectrogram
        """
        if random.random() > self.p:
            return spec
        
        # Handle both (H, W) and (C, H, W) formats
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)  # Add channel dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Clone to avoid modifying original
        augmented_spec = spec.clone()
        
        # Get dimensions: (channels, freq_bins, time_steps)
        _, freq_bins, time_steps = augmented_spec.shape
        
        # Apply frequency masking (horizontal bands)
        for _ in range(self.num_freq_masks):
            if self.freq_mask_param > 0 and freq_bins > self.freq_mask_param:
                # Random mask size and position
                mask_size = random.randint(0, min(self.freq_mask_param, freq_bins))
                if mask_size > 0:
                    mask_start = random.randint(0, freq_bins - mask_size)
                    augmented_spec[:, mask_start:mask_start + mask_size, :] = self.mask_value
        
        # Apply time masking (vertical bands)
        for _ in range(self.num_time_masks):
            if self.time_mask_param > 0 and time_steps > self.time_mask_param:
                # Random mask size and position
                mask_size = random.randint(0, min(self.time_mask_param, time_steps))
                if mask_size > 0:
                    mask_start = random.randint(0, time_steps - mask_size)
                    augmented_spec[:, :, mask_start:mask_start + mask_size] = self.mask_value
        
        if squeeze_output:
            augmented_spec = augmented_spec.squeeze(0)
        
        return augmented_spec

class PILSpecAugment:
    """
    SpecAugment for PIL Images (for use with torchvision transforms).
    
    This version works directly on PIL Images before tensor conversion,
    which is useful when integrating with existing transform pipelines.
    """
    
    def __init__(self, time_mask_param=40, freq_mask_param=15, 
                 num_time_masks=1, num_freq_masks=1, 
                 mask_value=0, p=0.8):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.mask_value = mask_value  # 0-255 for PIL
        self.p = p
    
    def __call__(self, img):
        """
        Apply SpecAugment to PIL Image.
        
        Args:
            img (PIL.Image): Input spectrogram image
        
        Returns:
            PIL.Image: Augmented spectrogram image
        """
        if random.random() > self.p:
            return img
        
        # Convert to numpy for easier manipulation
        img_array = np.array(img)
        
        # Get dimensions (height=freq_bins, width=time_steps)
        height, width = img_array.shape[:2]
        
        # Apply frequency masking (horizontal bands)
        for _ in range(self.num_freq_masks):
            if self.freq_mask_param > 0 and height > self.freq_mask_param:
                mask_size = random.randint(0, min(self.freq_mask_param, height))
                if mask_size > 0:
                    mask_start = random.randint(0, height - mask_size)
                    img_array[mask_start:mask_start + mask_size, :] = self.mask_value
        
        # Apply time masking (vertical bands)
        for _ in range(self.num_time_masks):
            if self.time_mask_param > 0 and width > self.time_mask_param:
                mask_size = random.randint(0, min(self.time_mask_param, width))
                if mask_size > 0:
                    mask_start = random.randint(0, width - mask_size)
                    img_array[:, mask_start:mask_start + mask_size] = self.mask_value
        
        return Image.fromarray(img_array)

def get_recommended_params(num_samples, num_classes, input_size=(224, 313)):
    """
    Get recommended SpecAugment parameters based on dataset characteristics.
    
    Args:
        num_samples (int): Total number of training samples
        num_classes (int): Number of classes
        input_size (tuple): (height, width) of spectrograms
    
    Returns:
        dict: Recommended parameters
    """
    height, width = input_size
    
    # Calculate samples per class
    samples_per_class = num_samples / num_classes
    
    # Adjust aggressiveness based on dataset size
    if samples_per_class < 50:
        # Small dataset - less aggressive
        time_mask_param = min(25, width // 12)
        freq_mask_param = min(10, height // 20)
        p = 0.6
        num_time_masks = 1
        num_freq_masks = 1
    elif samples_per_class < 100:
        # Medium dataset - moderate
        time_mask_param = min(40, width // 8)
        freq_mask_param = min(15, height // 15)
        p = 0.8
        num_time_masks = 1
        num_freq_masks = 1
    else:
        # Large dataset - more aggressive
        time_mask_param = min(50, width // 6)
        freq_mask_param = min(20, height // 12)
        p = 0.9
        num_time_masks = 2
        num_freq_masks = 1
    
    return {
        'time_mask_param': time_mask_param,
        'freq_mask_param': freq_mask_param,
        'num_time_masks': num_time_masks,
        'num_freq_masks': num_freq_masks,
        'mask_value': 0.0,  # Black masks for normalized spectrograms
        'p': p
    }

def visualize_specaugment(original_spec, augmented_spec, title="SpecAugment Comparison"):
    """
    Visualize original and augmented spectrograms side by side.
    
    Args:
        original_spec (torch.Tensor or np.ndarray): Original spectrogram
        augmented_spec (torch.Tensor or np.ndarray): Augmented spectrogram
        title (str): Plot title
    """
    # Convert tensors to numpy if needed
    if isinstance(original_spec, torch.Tensor):
        if original_spec.dim() == 3:  # (C, H, W)
            original_spec = original_spec.squeeze(0).numpy()
        else:
            original_spec = original_spec.numpy()
    
    if isinstance(augmented_spec, torch.Tensor):
        if augmented_spec.dim() == 3:  # (C, H, W)
            augmented_spec = augmented_spec.squeeze(0).numpy()
        else:
            augmented_spec = augmented_spec.numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original spectrogram
    im1 = ax1.imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title('Original Spectrogram')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Frequency Bins')
    plt.colorbar(im1, ax=ax1)
    
    # Augmented spectrogram
    im2 = ax2.imshow(augmented_spec, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title('SpecAugment Applied')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Frequency Bins')
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def test_specaugment_on_random_spec(shape=(224, 313), **augment_params):
    """
    Test SpecAugment on a random spectrogram for visualization.
    
    Args:
        shape (tuple): Shape of random spectrogram (height, width)
        **augment_params: Parameters for SpecAugment
    """
    # Create random spectrogram
    random_spec = torch.rand(shape)
    
    # Apply SpecAugment
    augmenter = SpecAugment(**augment_params)
    augmented_spec = augmenter(random_spec)
    
    # Visualize
    visualize_specaugment(random_spec, augmented_spec, "SpecAugment Test on Random Data")
    
    return random_spec, augmented_spec

class GaussianNoise(nn.Module):
    """
    Additive Gaussian noise augmentation for spectrograms.
    
    Args:
        std (float): Standard deviation of Gaussian noise
        p (float): Probability of applying noise
    """
    
    def __init__(self, std=0.02, p=0.8):
        super().__init__()
        self.std = std
        self.p = p
    
    def forward(self, spec):
        """
        Apply Gaussian noise to spectrogram.
        
        Args:
            spec (torch.Tensor): Input spectrogram
        
        Returns:
            torch.Tensor: Noisy spectrogram
        """
        if random.random() > self.p:
            return spec
        
        noise = torch.randn_like(spec) * self.std
        return torch.clamp(spec + noise, 0.0, 1.0)  # Clamp to valid range


class OnTheFlyAugmentation:
    """
    Combined SpecAugment and Gaussian noise augmentation for training.
    Applied only during training, not validation/testing.
    """
    
    def __init__(self, use_spec_augment=False, use_gaussian_noise=False, 
                 spec_augment_params=None, gaussian_noise_params=None, training=True):
        """
        Initialize augmentation pipeline.
        
        Args:
            use_spec_augment (bool): Whether to apply SpecAugment
            use_gaussian_noise (bool): Whether to apply Gaussian noise
            spec_augment_params (dict): Parameters for SpecAugment
            gaussian_noise_params (dict): Parameters for Gaussian noise
            training (bool): Whether in training mode
        """
        self.use_spec_augment = use_spec_augment
        self.use_gaussian_noise = use_gaussian_noise
        self.training = training
        
        # Initialize SpecAugment if enabled
        if self.use_spec_augment and self.training:
            if spec_augment_params is None:
                spec_augment_params = get_recommended_params(1000, 31)  # Default params
            self.spec_augment = SpecAugment(**spec_augment_params)
        else:
            self.spec_augment = None
        
        # Initialize Gaussian noise if enabled
        if self.use_gaussian_noise and self.training:
            if gaussian_noise_params is None:
                gaussian_noise_params = {'std': 0.02, 'p': 0.8}
            self.gaussian_noise = GaussianNoise(**gaussian_noise_params)
        else:
            self.gaussian_noise = None
    
    def __call__(self, spec):
        """
        Apply augmentations in sequence: SpecAugment -> Gaussian noise.
        
        Args:
            spec (torch.Tensor): Input spectrogram
        
        Returns:
            torch.Tensor: Augmented spectrogram
        """
        if not self.training:
            return spec
        
        # Apply SpecAugment first
        if self.spec_augment is not None:
            spec = self.spec_augment(spec)
        
        # Apply Gaussian noise second
        if self.gaussian_noise is not None:
            spec = self.gaussian_noise(spec)
        
        return spec
    
    def set_training(self, training):
        """Set training mode."""
        self.training = training


def get_augmentation_params(dataset_size, num_classes, aggressive=False):
    """
    Get recommended augmentation parameters.
    
    Args:
        dataset_size (int): Total training samples
        num_classes (int): Number of classes
        aggressive (bool): Whether to use aggressive augmentation
    
    Returns:
        dict: Combined parameters for both augmentations
    """
    # Get SpecAugment params
    spec_params = get_recommended_params(dataset_size, num_classes)
    
    # Adjust for aggressiveness
    if aggressive:
        spec_params['p'] = min(0.95, spec_params['p'] + 0.1)
        gaussian_params = {'std': 0.03, 'p': 0.9}
    else:
        gaussian_params = {'std': 0.02, 'p': 0.8}
    
    return {
        'spec_augment_params': spec_params,
        'gaussian_noise_params': gaussian_params
    }

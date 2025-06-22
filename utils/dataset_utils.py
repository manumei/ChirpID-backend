# Dataset and Data Loading Utilities
# Handles PyTorch dataset creation, data standardization, and augmentation

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image
import os
from utils.specaugment import SpecAugment, PILSpecAugment, get_recommended_params

# Custom Dataset Classes
class StandardizedDataset(Dataset):
    """Dataset class for standardized data (needed for multiprocessing compatibility)."""
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class StandardizedSubset(Dataset):
    """Dataset subset with on-the-fly standardization using training statistics."""
    def __init__(self, original_dataset, indices, mean, std):
        self.dataset = original_dataset
        self.indices = indices
        self.mean = mean
        self.std = std + 1e-8  # Add epsilon to avoid division by zero
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.dataset[real_idx]
        x_standardized = (x - self.mean) / self.std
        return x_standardized, y

class FastStandardizedSubset(Dataset):
    """Optimized standardized subset class for faster training."""
    def __init__(self, original_dataset, indices, mean, std):
        self.dataset = original_dataset
        self.indices = indices
        self.mean = mean
        self.std = std + 1e-8
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.dataset[real_idx]
        x_standardized = (x - self.mean) / self.std
        return x_standardized, y

class AugmentedDataset(Dataset):
    """Dataset wrapper that applies SpecAugment during training."""
    def __init__(self, base_dataset, augment_params=None, training=True):
        self.base_dataset = base_dataset
        self.training = training
        
        if augment_params is None:
            augment_params = get_recommended_params()
        
        self.augment_params = augment_params
        
        # Initialize SpecAugment
        if training:
            self.spec_augment = PILSpecAugment(
                freq_mask_param=augment_params.get('freq_mask_param', 15),
                time_mask_param=augment_params.get('time_mask_param', 35),
                num_freq_masks=augment_params.get('num_freq_masks', 1),
                num_time_masks=augment_params.get('num_time_masks', 1)
            )
        else:
            self.spec_augment = None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        
        if self.training and self.spec_augment is not None:
            # Convert tensor to PIL Image, apply augmentation, convert back
            if isinstance(x, torch.Tensor):
                # Convert tensor to numpy array
                x_np = x.squeeze().numpy() if x.dim() > 2 else x.numpy()
                # Normalize to 0-255 range for PIL
                x_normalized = ((x_np - x_np.min()) / (x_np.max() - x_np.min()) * 255).astype(np.uint8)
                # Convert to PIL Image
                x_pil = Image.fromarray(x_normalized)
                # Apply augmentation
                x_aug = self.spec_augment(x_pil)
                # Convert back to tensor
                x = torch.tensor(np.array(x_aug), dtype=torch.float32).unsqueeze(0)
                
        return x, y

# Data Loading Functions
def get_spect_matrix_list(spects_source_dir, spects_meta_df):
    """
    Load spectrograms directly as matrices without flattening to CSV.
    
    Args:
        spects_source_dir (str): Directory where spectrogram images are stored in .png format
        spects_meta_df (pd.DataFrame): DataFrame with columns 'filename', 'class_id', and 'author'
    
    Returns:
        tuple: (matrices_list, labels_list, authors_list)
    """
    matrices_list = []
    labels_list = []
    authors_list = []
    
    spects_meta_df = spects_meta_df.dropna(subset=['filename', 'class_id', 'author'])

    print(f"Processing {len(spects_meta_df)} spectrograms...")
    processed_count = 0
    skipped_count = 0

    for _, row in spects_meta_df.iterrows():
        filename = row['filename']
        class_id = row['class_id']
        author = row['author']

        image_path = os.path.join(spects_source_dir, filename)
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            skipped_count += 1
            continue

        img = Image.open(image_path).convert('L')  # Ensure grayscale
        
        expected_shape = (313, 224)  # PIL uses (width, height) format
        if img.size != expected_shape:
            print(f"Warning: Unexpected image size: {img.size} in file {image_path}. Expected {expected_shape}.")
            # Resize if needed
            img = img.resize(expected_shape)

        # Convert to numpy array (this gives us height x width, i.e., 313 x 224)
        pixels = np.array(img)
        
        matrices_list.append(pixels)
        labels_list.append(class_id)
        authors_list.append(author)
        processed_count += 1

    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")

    if not matrices_list:
        raise ValueError("No spectrograms were loaded. Check paths and metadata consistency.")

    return matrices_list, labels_list, authors_list

def create_augmented_dataset_wrapper(dataset, augment_params=None, training=True):
    """
    Create an augmented dataset wrapper with SpecAugment.
    
    Args:
        dataset: Base PyTorch dataset
        augment_params: Dictionary of SpecAugment parameters
        training: Whether this is for training (applies augmentation) or validation
    
    Returns:
        AugmentedDataset wrapper
    """
    return AugmentedDataset(dataset, augment_params, training)

# Standardization Utilities
def compute_standardization_stats(dataset, indices, sample_size=1000):
    """
    Efficiently compute standardization statistics from a dataset.
    
    Args:
        dataset: PyTorch dataset
        indices: Indices to sample from
        sample_size: Number of samples to use for computing statistics
    
    Returns:
        tuple: (mean, std)
    """
    # Use a subset for efficient computation
    sample_size = min(sample_size, len(indices))
    sample_indices = np.random.choice(indices, sample_size, replace=False)
    
    # Batch computation for efficiency
    sample_tensors = []
    batch_size = 32
    
    for i in range(0, len(sample_indices), batch_size):
        batch_indices = sample_indices[i:i+batch_size]
        batch_data = torch.stack([dataset[idx][0] for idx in batch_indices])
        sample_tensors.append(batch_data)
    
    sample_data = torch.cat(sample_tensors, dim=0)
    mean = sample_data.mean()
    std = sample_data.std() + 1e-8  # Add epsilon to avoid division by zero
    
    return mean, std

def create_standardized_subset(dataset, indices, mean, std, fast=False):
    """
    Create a standardized subset of the dataset.
    
    Args:
        dataset: Original PyTorch dataset
        indices: Indices for the subset
        mean: Mean for standardization
        std: Standard deviation for standardization
        fast: Whether to use the fast implementation
    
    Returns:
        Standardized subset
    """
    if fast:
        return FastStandardizedSubset(dataset, indices, mean, std)
    else:
        return StandardizedSubset(dataset, indices, mean, std)

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
    """Worker-safe dataset subset with on-the-fly standardization using training statistics."""
    def __init__(self, original_dataset, indices, mean, std):
        self.dataset = original_dataset
        self.indices = list(indices)  # Convert to list for pickling
        self.mean = float(mean)  # Store as primitive types for worker safety
        self.std = float(std + 1e-8)  # Add epsilon to avoid division by zero
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.dataset[real_idx]
        # Apply standardization with primitive operations only
        x_standardized = (x - self.mean) / self.std
        return x_standardized, y

class FastStandardizedSubset(Dataset):
    """Worker-safe optimized standardized subset class for faster training."""
    def __init__(self, original_dataset, indices, mean, std):
        self.dataset = original_dataset
        self.indices = list(indices)  # Convert to list for pickling
        self.mean = float(mean)  # Store as primitive types for worker safety
        self.std = float(std + 1e-8)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.dataset[real_idx]
        # Apply standardization with primitive operations only
        x_standardized = (x - self.mean) / self.std
        return x_standardized, y

class AugmentedDataset(Dataset):
    """Worker-safe dataset wrapper that applies on-the-fly augmentation during training."""
    def __init__(self, base_dataset, use_spec_augment=False, use_gaussian_noise=False, 
                augment_params=None, training=True):
        self.base_dataset = base_dataset
        self.training = training
        self.use_spec_augment = use_spec_augment
        self.use_gaussian_noise = use_gaussian_noise
        
        # Store augmentation parameters as primitive types for worker safety
        if augment_params is None and (use_spec_augment or use_gaussian_noise):
            # Import here to avoid circular imports
            from utils.specaugment import get_augmentation_params
            augment_params = get_augmentation_params(len(base_dataset), 31)
        
        # Store parameters as simple dictionaries (picklable)
        self.spec_augment_params = augment_params.get('spec_augment_params') if augment_params else None
        self.gaussian_noise_params = augment_params.get('gaussian_noise_params') if augment_params else None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        
        # Apply augmentation only during training
        if self.training:
            # Import augmentation classes inside __getitem__ to avoid worker issues
            from utils.specaugment import SpecAugment, GaussianNoise
            
            # Apply SpecAugment if enabled
            if self.use_spec_augment and self.spec_augment_params:
                spec_augment = SpecAugment(**self.spec_augment_params)
                x = spec_augment(x)
            
            # Apply Gaussian noise if enabled
            if self.use_gaussian_noise and self.gaussian_noise_params:
                gaussian_noise = GaussianNoise(**self.gaussian_noise_params)
                x = gaussian_noise(x)
        
        return x, y
    
    def set_training(self, training):
        """Set training mode for augmentation."""
        self.training = training

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
            raise ValueError(f"Image {filename} has unexpected shape {img.size}. Expected {expected_shape}.")

        # Convert to numpy array (this gives us height x width, i.e., (224, 313))
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

def create_augmented_dataset_wrapper(dataset, use_spec_augment=False, use_gaussian_noise=False, 
                                    augment_params=None, training=True):
    """
    Create an augmented dataset wrapper with SpecAugment and/or Gaussian noise.
    
    Args:
        dataset: Base PyTorch dataset
        use_spec_augment (bool): Whether to apply SpecAugment
        use_gaussian_noise (bool): Whether to apply Gaussian noise
        augment_params: Dictionary of augmentation parameters
        training: Whether this is for training (applies augmentation) or validation
    
    Returns:
        AugmentedDataset wrapper
    """
    return AugmentedDataset(dataset, use_spec_augment, use_gaussian_noise, augment_params, training)

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

import os, sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
else:
    print("⚠️  CUDA not available - running on CPU (will be slow)")

# Performance optimization settings
ENABLE_OPTIMIZATIONS = True  # Set to False to disable all optimizations
ENABLE_PARALLEL_FOLDS = False  # Set to True for cross-validation mode
MAX_PARALLEL_FOLDS = -1  # Adjust based on GPU memory

def load_npy_data(specs_dir: str, specs_csv_path: str) -> Tuple[np.ndarray, np.array, np.array]:
    """
    Load spectrograms from .npy files and metadata from CSV.
    
    Args:
        specs_dir (str): Directory containing .npy spectrogram files
        specs_csv_path (str): Path to CSV file containing metadata (filename, class_id, author)
    
    Returns:
        Tuple[np.ndarray, np.array, np.array]: Returns features, labels, and authors.
        Features are already normalized to [0,1] and shaped as (N, 1, 224, 313)
    """
    # Load metadata CSV
    df = pd.read_csv(specs_csv_path)
    
    print(f"Metadata shape: {df.shape}")
    print(f"Number of classes: {df['class_id'].nunique()}")
    print(f"Number of authors: {df['author'].nunique()}")
    
    # Extract labels and authors
    labels = df['class_id'].values.astype(np.int64)
    authors = df['author'].values
    filenames = df['filename'].values
    
    # Load spectrograms from .npy files
    features_list = []
    valid_indices = []
    
    for i, filename in enumerate(filenames):
        spec_path = os.path.join(specs_dir, filename)
        
        if os.path.exists(spec_path):
            try:
                # Load .npy file - already normalized to [0,1] as float32
                spec_array = np.load(spec_path)
                
                # Add channel dimension: (1, height, width)
                spec_array = spec_array[np.newaxis, ...]
                
                features_list.append(spec_array)
                valid_indices.append(i)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"File not found: {spec_path}")
    
    # Convert to numpy array
    features = np.array(features_list, dtype=np.float32)
    
    # Filter labels and authors to match loaded features
    labels = labels[valid_indices]
    authors = authors[valid_indices]
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Authors shape: {authors.shape}")
    print(f"Unique classes: {len(np.unique(labels))}")
    print(f"Unique authors: {len(np.unique(authors))}")
    print(f"Successfully loaded {len(features)} out of {len(filenames)} spectrograms")
    
    # Clean up
    del df
    
    return features, labels, authors

specs_dir = os.path.join('..', 'database', 'specs')
specs_csv_path = os.path.join('..', 'database', 'meta', 'final_specs.csv')

# Adjust paths when running from utils directory
if not os.path.exists(specs_csv_path):
    specs_dir = os.path.join('database', 'specs')
    specs_csv_path = os.path.join('database', 'meta', 'final_specs.csv')
features, labels, authors = load_npy_data(specs_dir, specs_csv_path)

# Split with a set seed
from utils.split import get_set_seed_indices, get_set_seed_kfold_indices, display_split_statistics
seed_single = 245323 # Quality: 0.2671
seed_kfold = 11052 # Quality: 0.3332

single_fold_split = get_set_seed_indices(
    features=features,
    labels=labels, 
    authors=authors,
    test_size=0.2,
    seed=seed_single)

kfold_splits = get_set_seed_kfold_indices(
    features=features,
    labels=labels,
    authors=authors,
    n_splits=4,
    seed=seed_kfold)

display_split_statistics(single_fold_split, "single")
display_split_statistics(kfold_splits, "kfold")

class_num = len(np.unique(labels))
config_a = {
    'name': 'Parameters Frankenstein',
    'use_adam': True,
    'estop_thresh': 36,
    'batch_size': 40,
    'use_class_weights': True,
    'l2_regularization': 0.0003,
    'lr_schedule': {'type': 'exponential', 'gamma': 0.97},
    'initial_lr': 0.0024, # also try 0.00137
    'standardize': True,
    'spec_augment': True,
    'noise_augment': False,
    'num_epochs': 220,
    'mixed_precision': True,
    'gradient_clipping': 1.0,
    'parallel_folds': False,
    'max_parallel_folds': 2,
    'optimize_dataloaders': True,
}

from utils.training_core import single_fold_training
from utils.models import (
    BirdCNN_v1, BirdCNN_v2, BirdCNN_v3, BirdCNN_v4, BirdCNN_v5, BirdCNN_v6, BirdCNN_v7, BirdCNN_v8,
    BirdCNN_v9, BirdCNN_v10, BirdCNN_v11, BirdCNN_v12, BirdCNN_v13, BirdCNN_v14, BirdCNN_v15, BirdCNN_v16
)

if __name__ == "__main__":
    print("Running model testing script...")
    debugging_models = [BirdCNN_v4, BirdCNN_v8, BirdCNN_v13]

    for model_class in debugging_models:
        result = single_fold_training(
            features=features,
            labels=labels,
            authors=authors,
            model_class=model_class,
            num_classes=class_num,
            config=config_a,
            spec_augment=config_a['spec_augment'],
            gaussian_noise=config_a['noise_augment'],
            precomputed_split=single_fold_split,  # Use pre-computed single fold split
            config_id="Config A"  # Pass config_id for progress bar
        )

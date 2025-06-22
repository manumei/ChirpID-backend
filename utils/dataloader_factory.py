# DataLoader Factory - Optimal DataLoader Configuration
# Provides hardware-optimized DataLoader configurations for RTX 5080 + Ryzen 9 7950X

import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np


def worker_init_fn(worker_id):
    """Initialize random state per worker for consistent augmentation."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


class OptimalDataLoaderFactory:
    """Factory for creating optimized DataLoaders based on hardware and dataset characteristics."""
    
    @staticmethod
    def get_optimal_config(dataset_size, has_augmentation=False, has_standardization=False):
        """
        Get optimal DataLoader configuration for current hardware.
        
        Hardware-specific settings for RTX 5080 + Ryzen 9 7950X:
        - RTX 5080: High VRAM allows aggressive prefetching and pin_memory
        - Ryzen 9 7950X: 32 logical cores support 12-16 workers efficiently
        
        Args:
            dataset_size (int): Size of the dataset
            has_augmentation (bool): Whether the dataset includes augmentation
            has_standardization (bool): Whether the dataset includes standardization
            
        Returns:
            dict: Optimal DataLoader configuration
        """
        # Hardware-specific base settings
        base_config = {
            'pin_memory': torch.cuda.is_available(),  # RTX 5080 has high VRAM
            'persistent_workers': True,  # Reduce spawn overhead
            'drop_last': True,  # Consistent batch sizes
        }
        
        # Determine worker count based on operations complexity
        if has_augmentation or has_standardization:
            # Use fewer workers for complex operations to avoid bottlenecks
            base_config['num_workers'] = 8
            base_config['prefetch_factor'] = 4
        else:
            # Use more workers for simple tensor loading
            base_config['num_workers'] = 12
            base_config['prefetch_factor'] = 6
            
        # Adjust for dataset size
        if dataset_size < 1000:
            base_config['num_workers'] = min(base_config['num_workers'], 4)
        elif dataset_size > 10000:
            base_config['num_workers'] = min(base_config['num_workers'] + 2, 16)
              # Disable persistent workers if num_workers is 0
        if base_config['num_workers'] == 0:
            base_config['persistent_workers'] = False
            base_config.pop('prefetch_factor', None)
        
        return base_config
    
    @staticmethod
    def create_training_loader(dataset, batch_size, **kwargs):
        """
        Create optimized training DataLoader.
        
        Args:
            dataset: PyTorch dataset
            batch_size (int): Batch size
            **kwargs: Additional configuration overrides
            
        Returns:
            DataLoader: Optimized training DataLoader
        """
        config = OptimalDataLoaderFactory.get_optimal_config(
            len(dataset), 
            kwargs.get('has_augmentation', False),
            kwargs.get('has_standardization', False)
        )
        
        # Remove our custom flags from kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['has_augmentation', 'has_standardization']}
        
        # Update with user overrides
        config.update(clean_kwargs)
        config['shuffle'] = True        # Add worker initialization for augmentation
        if kwargs.get('has_augmentation', False) and config['num_workers'] > 0:
            config['worker_init_fn'] = worker_init_fn
        
        # Ensure compatibility when num_workers is 0
        if config['num_workers'] == 0:
            config.pop('prefetch_factor', None)
            config['persistent_workers'] = False  # Must be False when num_workers is 0
            
        return DataLoader(dataset, batch_size=batch_size, **config)
    
    @staticmethod  
    def create_validation_loader(dataset, batch_size, **kwargs):
        """
        Create optimized validation DataLoader.
        
        Args:
            dataset: PyTorch dataset
            batch_size (int): Batch size
            **kwargs: Additional configuration overrides
            
        Returns:
            DataLoader: Optimized validation DataLoader
        """
        config = OptimalDataLoaderFactory.get_optimal_config(
            len(dataset),
            has_augmentation=False,  # No augmentation in validation
            has_standardization=kwargs.get('has_standardization', False)
        )
        
        # Remove our custom flags from kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['has_augmentation', 'has_standardization']}        # Update with user overrides
        config.update(clean_kwargs)
        config['shuffle'] = False
        
        # Ensure compatibility when num_workers is 0
        if config['num_workers'] == 0:
            config.pop('prefetch_factor', None)
            config['persistent_workers'] = False  # Must be False when num_workers is 0
        
        return DataLoader(dataset, batch_size=batch_size, **config)
    
    @staticmethod
    def create_test_loader(dataset, batch_size, **kwargs):
        """
        Create optimized test DataLoader.
        
        Args:
            dataset: PyTorch dataset
            batch_size (int): Batch size
            **kwargs: Additional configuration overrides
            
        Returns:
            DataLoader: Optimized test DataLoader
        """
        return OptimalDataLoaderFactory.create_validation_loader(dataset, batch_size, **kwargs)


class DataLoaderConfigLogger:
    """Helper class to log DataLoader configurations for debugging."""
    
    @staticmethod
    def log_config(loader_type, config, dataset_size):
        """Log DataLoader configuration."""
        print(f"\\n{loader_type} DataLoader Configuration:")
        print(f"  Dataset size: {dataset_size}")
        print(f"  Workers: {config.get('num_workers', 0)}")
        print(f"  Pin memory: {config.get('pin_memory', False)}")
        print(f"  Persistent workers: {config.get('persistent_workers', False)}")
        print(f"  Prefetch factor: {config.get('prefetch_factor', 'N/A')}")
        print(f"  Batch size: {config.get('batch_size', 'N/A')}")

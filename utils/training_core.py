# Training Core - Top-Level Training Functions
# Provides the main training interface with clean, hierarchical structure

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from utils.training_engine import TrainingEngine
from utils.split import search_best_group_seed, search_best_group_seed_kfold
from utils.data_preparation import prepare_training_data, create_metadata_dataframe


def cross_val_training(data_path=None, features=None, labels=None, authors=None, 
                      model_class=None, num_classes=None, config=None,
                      spec_augment=False, gaussian_noise=False):
    """
    Top-level function for K-fold cross-validation training.
    
    Args:
        data_path (str, optional): Path to CSV file with training data
        features (np.ndarray, optional): Pre-loaded feature array  
        labels (np.ndarray, optional): Pre-loaded labels array
        authors (np.ndarray, optional): Pre-loaded authors array
        model_class: PyTorch model class to train
        num_classes (int): Number of output classes
        config (dict, optional): Training configuration parameters
        spec_augment (bool): Whether to apply SpecAugment during training
        gaussian_noise (bool): Whether to apply Gaussian noise during training
    
    Returns:
        dict: Complete training results including fold results and summary
    """
    print("=" * 60)
    print("CROSS-VALIDATION TRAINING")
    print("=" * 60)      # Set default configuration
    default_config = {
        'k_folds': 4,
        'num_epochs': 220,
        'batch_size': 24,
        'learning_rate': 0.001,
        'use_class_weights': False,
        'early_stopping': 35,
        'standardize': True,
        'aggregate_predictions': True,
        'max_split_attempts': 30000,
        'min_val_segments': 0,
        'spec_augment': spec_augment,        'gaussian_noise': gaussian_noise,
        # New optimization settings
        'optimize_dataloaders': True,
        'debug_dataloaders': False,  # Set to True for debugging
        'benchmark_performance': False,  # Set to True for performance testing
        # Mixed precision and gradient clipping optimizations
        'mixed_precision': True,  # Enable AMP for RTX 5080
        'gradient_clipping': 1.0,  # Gradient clipping value (0 to disable)
        # Parallel fold training optimization
        'parallel_folds': False,  # Enable parallel fold training (experimental)
        'max_parallel_folds': 2,  # Max concurrent folds (adjust for GPU memory)
    }
    config = {**default_config, **(config or {})}
    
    # Prepare training data
    features, labels, authors = prepare_training_data(data_path, features, labels, authors)
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    
    # Create metadata for splitting
    metadata_df = create_metadata_dataframe(labels, authors)
    
    # Find optimal k-fold splits with author grouping
    print(f"Finding best {config['k_folds']}-fold split with author grouping...")
    best_folds, best_score, best_seed = search_best_group_seed_kfold(
        df=metadata_df,
        max_attempts=config['max_split_attempts'],
        min_val_segments=config['min_val_segments'],
        n_splits=config['k_folds']
    )
    
    print(f"Best fold configuration found with seed {best_seed}")
    print(f"Average stratification score: {best_score:.3f}")
    
    # Convert fold indices for dataset
    fold_indices = []
    for train_df, val_df in best_folds:
        train_indices = train_df['sample_idx'].values
        val_indices = val_df['sample_idx'].values
        fold_indices.append((train_indices, val_indices))
    
    # Initialize training engine
    engine = TrainingEngine(
        model_class=model_class,
        num_classes=num_classes,
        config=config
    )
      # Execute cross-validation training (sequential or parallel)
    if config.get('parallel_folds', False):
        print(f"Using PARALLEL fold training (max {config['max_parallel_folds']} concurrent folds)")
        results, best_results = engine.run_cross_validation_parallel(
            dataset, fold_indices, config['max_parallel_folds']
        )
    else:
        print("Using SEQUENTIAL fold training")
        results, best_results = engine.run_cross_validation(dataset, fold_indices)
    
    print("\\n" + "=" * 60)
    print("CROSS-VALIDATION TRAINING COMPLETED")
    print("=" * 60)
    
    return results, best_results


def single_fold_training(data_path=None, features=None, labels=None, authors=None,
                        model_class=None, num_classes=None, config=None, 
                        use_predefined_split=True, spec_augment=False, gaussian_noise=False):
    """
    Top-level function for single fold training.
    
    Args:
        data_path (str, optional): Path to CSV file with training data
        features (np.ndarray, optional): Pre-loaded feature array
        labels (np.ndarray, optional): Pre-loaded labels array  
        authors (np.ndarray, optional): Pre-loaded authors array
        model_class: PyTorch model class to train
        num_classes (int): Number of output classes
        config (dict, optional): Training configuration parameters
        use_predefined_split (bool): Whether to use author-grouped splitting
        spec_augment (bool): Whether to apply SpecAugment during training
        gaussian_noise (bool): Whether to apply Gaussian noise during training
    
    Returns:
        dict: Complete training results
    """
    print("=" * 60)
    print("SINGLE FOLD TRAINING")
    print("=" * 60)
    
    # Set default configuration
    default_config = {
        'num_epochs': 250,
        'batch_size': 48,
        'learning_rate': 0.001,
        'use_class_weights': False,
        'early_stopping': 35,
        'standardize': True,
        'test_size': 0.2,
        'random_state': 435,
        'max_split_attempts': 10000,
        'min_test_segments': 5,
        'spec_augment': spec_augment,
        'gaussian_noise': gaussian_noise,
        # New optimization settings
        'optimize_dataloaders': True,
        'debug_dataloaders': False,  # Set to True for debugging
        'benchmark_performance': False,  # Set to True for performance testing
        # Additional training parameters
        'use_adam': True,  # Whether to use Adam optimizer
        'l2_regularization': 1e-4,  # L2 regularization strength
        'lr_schedule': None  # Learning rate schedule configuration
    }
    config = {**default_config, **(config or {})}
    
    # Prepare training data
    features, labels, authors = prepare_training_data(data_path, features, labels, authors)
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    
    # Initialize training engine
    engine = TrainingEngine(
        model_class=model_class,
        num_classes=num_classes,
        config=config
    )
    
    if use_predefined_split:
        # Create metadata for splitting
        metadata_df = create_metadata_dataframe(labels, authors)
        
        # Find optimal 80-20 split with author grouping
        print(f"Finding best {int((1-config['test_size'])*100)}-{int(config['test_size']*100)} split with author grouping...")
        dev_df, test_df, best_split_score = search_best_group_seed(
            df=metadata_df,
            test_size=config['test_size'],
            max_attempts=config['max_split_attempts'],
            min_test_segments=config['min_test_segments']
        )
        
        # Extract indices
        train_indices = dev_df['sample_idx'].values
        val_indices = test_df['sample_idx'].values
        
        print(f"Best split found with score: {best_split_score:.3f}")
        print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
        
        # Execute training with predefined split
        results = engine.run_single_fold_predefined(dataset, train_indices, val_indices)
        
    else:
        # Execute training with stratified split
        results = engine.run_single_fold_stratified(dataset)
        
    print("\\n" + "=" * 60)
    print("SINGLE FOLD TRAINING COMPLETED")
    print("=" * 60)
    
    return results
    """
    Top-level function for single-fold training with optimized DataLoaders.
    
    Args:
        data_path (str, optional): Path to CSV file with training data
        features (np.ndarray, optional): Pre-loaded feature array  
        labels (np.ndarray, optional): Pre-loaded labels array
        authors (np.ndarray, optional): Pre-loaded authors array
        model_class: PyTorch model class to train
        num_classes (int): Number of output classes
        config (dict, optional): Training configuration parameters
        spec_augment (bool): Whether to apply SpecAugment during training
        gaussian_noise (bool): Whether to apply Gaussian noise during training
    
    Returns:
        dict: Complete training results
    """
    print("=" * 60)
    print("SINGLE-FOLD TRAINING")
    print("=" * 60)
    
    # Set default configuration
    default_config = {
        'num_epochs': 220,
        'batch_size': 24,
        'learning_rate': 0.001,
        'use_class_weights': False,
        'early_stopping': 35,
        'standardize': True,
        'test_size': 0.2,
        'random_state': 42,
        'spec_augment': spec_augment,
        'gaussian_noise': gaussian_noise,
        # New optimization settings        'optimize_dataloaders': True,
        'debug_dataloaders': False,
        'benchmark_performance': False,
        # Mixed precision and gradient clipping optimizations
        'mixed_precision': True,  # Enable AMP for RTX 5080
        'gradient_clipping': 1.0,  # Gradient clipping value (0 to disable)
    }
    config = {**default_config, **(config or {})}
    
    # Prepare training data
    features, labels, authors = prepare_training_data(data_path, features, labels, authors)
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    
    # Create metadata for splitting
    metadata_df = create_metadata_dataframe(labels, authors)
    
    # Find optimal train/validation split with author grouping
    print(f"Finding best train/validation split with author grouping...")
    from utils.split import search_best_group_seed
    
    train_df, val_df, best_seed = search_best_group_seed(
        df=metadata_df,
        test_size=config['test_size'],
        max_attempts=config.get('max_split_attempts', 30000),
        min_val_segments=config.get('min_val_segments', 0)
    )
    
    print(f"Best split found with seed {best_seed}")
    
    # Convert to indices
    train_indices = train_df['sample_idx'].values
    val_indices = val_df['sample_idx'].values
    
    # Initialize training engine
    engine = TrainingEngine(
        model_class=model_class,
        num_classes=num_classes,
        config=config
    )
    
    # Execute single-fold training
    results = engine._train_single_fold_with_indices(
        dataset, train_indices, val_indices, fold_num=1
    )
    
    print("\\n" + "=" * 60)
    print("SINGLE-FOLD TRAINING COMPLETED")
    print("=" * 60)
    
    return results

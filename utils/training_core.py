# Training Core - Top-Level Training Functions
# Provides the main training interface with clean, hierarchical structure
# 
# NEW OPTIMIZATION: Added support for pre-computed splits to eliminate
# redundant split computation during hyperparameter sweeping.
# 
# Functions now accept precomputed_splits/precomputed_split parameters
# to reuse previously computed author-grouped splits across multiple
# training configurations, significantly improving efficiency.

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
                        spec_augment=False, gaussian_noise=False, 
                        precomputed_splits=None, config_id=None):
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
        precomputed_splits (tuple, optional): Pre-computed (fold_indices, best_score, best_seed) 
                                            from split generation to avoid recomputation
    
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
        'parallel_folds': False,  # Enable parallel fold training (always uses GPU parallel)
        'max_parallel_folds': 3,  # Max concurrent folds (adjust for GPU memory)
    }
    config = {**default_config, **(config or {})}
    
    # Add config_id if provided
    if config_id is not None:
        config['config_id'] = config_id
    
    # Prepare training data
    features, labels, authors = prepare_training_data(data_path, features, labels, authors)
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    
    # Use precomputed splits if available, otherwise compute them
    if precomputed_splits is not None:
        fold_indices, best_score, best_seed = precomputed_splits
        print(f"Using precomputed {config['k_folds']}-fold splits (seed {best_seed}, score {best_score:.3f})")
    else:
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
        print(f"Using TRUE GPU PARALLEL fold training (max {config['max_parallel_folds']} concurrent folds)")
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
                        use_predefined_split=True, spec_augment=False, gaussian_noise=False,
                        precomputed_split=None, config_id=None):
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
        precomputed_split (tuple, optional): Pre-computed (train_indices, val_indices, best_score) 
                                           from split generation to avoid recomputation
    
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
    
    # Add config_id if provided
    if config_id is not None:
        config['config_id'] = config_id
    
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
        # Use precomputed split if available, otherwise compute it
        if precomputed_split is not None:
            train_indices, val_indices, best_split_score, best_split_seed = precomputed_split
            print(f"Using precomputed split with score: {best_split_score:.3f}")
            print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
        else:
            # Create metadata for splitting
            metadata_df = create_metadata_dataframe(labels, authors)
            
            # Find optimal 80-20 split with author grouping
            print(f"Finding best {int((1-config['test_size'])*100)}-{int(config['test_size']*100)} split with author grouping...")
            dev_df, test_df, best_split_score, best_seed = search_best_group_seed(
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

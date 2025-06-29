# Cross-Validation Utilities
# Legacy support for cross-validation - use training_core.cross_val_training() instead

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

from utils.dataset_utils import (
    StandardizedDataset, compute_standardization_stats, 
    create_standardized_subset
)
from utils.training_utils import train_single_fold, validate_epoch
from utils.dataloader_factory import OptimalDataLoaderFactory


def k_fold_cross_validation_with_predefined_folds(dataset, fold_indices, model_class, num_classes, 
                                                    num_epochs=300, batch_size=32, lr=0.001, 
                                                    aggregate_predictions=True, use_class_weights=True, 
                                                    estop=35, standardize=False):
    """
    Legacy function - use training_core.cross_val_training() instead.
    Perform K-Fold Cross Validation training with predefined fold indices.
    """
    print("⚠️  Warning: This function is deprecated. Use training_core.cross_val_training() instead.")
    
    # Import here to avoid circular imports
    from utils.training_engine import TrainingEngine
    
    config = {
        'k_folds': len(fold_indices),
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'aggregate_predictions': aggregate_predictions,
        'use_class_weights': use_class_weights,
        'early_stopping': estop,
        'standardize': standardize
    }
    
    engine = TrainingEngine(model_class, num_classes, config)
    return engine.run_cross_validation(dataset, fold_indices)


def k_fold_cross_validation(dataset, model_class, num_classes, k_folds=4, 
                           num_epochs=300, batch_size=32, lr=0.001, random_state=435, 
                           aggregate_predictions=True, use_class_weights=True, estop=35,
                           standardize=False):
    """
    Legacy function - use training_core.cross_val_training() instead.
    Perform K-Fold Cross Validation training with F1 score reporting and early stopping.
    """
    print("⚠️  Warning: This function is deprecated. Use training_core.cross_val_training() instead.")
    
    # For backward compatibility, we'll still provide the full implementation
    # but users should migrate to the new API
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract all labels for stratification
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]    
    # Check if we have enough samples per class for k-fold CV
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    min_samples_per_class = min(label_counts)
    
    if min_samples_per_class < k_folds:
        print(f"WARNING: Some classes have fewer than {k_folds} samples (minimum: {min_samples_per_class})")
        print("This may cause issues with stratified k-fold CV. Consider reducing k_folds or collecting more data.")
        
    # Simple stratified k-fold - use new API for production
    skfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    fold_splits = list(skfold.split(range(len(dataset)), all_labels))
    
    # Import and use new training engine
    from utils.training_engine import TrainingEngine
    
    config = {
        'k_folds': k_folds,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'aggregate_predictions': aggregate_predictions,
        'use_class_weights': use_class_weights,
        'early_stopping': estop,
        'standardize': standardize
    }
    
    engine = TrainingEngine(model_class, num_classes, config)
    return engine.run_cross_validation(dataset, fold_splits)

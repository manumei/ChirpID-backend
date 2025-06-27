# Utility functions that haven't been moved to specialized modules
# This file should be minimized - most functions have been moved to:
# - data_processing.py: Audio and spectrogram processing
# - training_utils.py: Model training functions
# - evaluation_utils.py: Model evaluation and visualization  
# - cross_validation.py: K-fold cross-validation
# - dataset_utils.py: Dataset classes and utilities

import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils.specaugment import SpecAugment, PILSpecAugment, get_recommended_params, visualize_specaugment
from utils.dataset_utils import StandardizedDataset, StandardizedSubset, FastStandardizedSubset

# Import functions from specialized modules for backward compatibility
from utils.data_processing import (
    clean_dir, count_files_in_dir, lbrs_loading, get_rmsThreshold, 
    reduce_noise_seg, get_spec_norm, get_spec_image, get_spec_npy, save_test_audios,
    save_audio_segments_to_disk, load_audio_segments_from_disk,
    load_audio_files, calculate_class_totals, extract_balanced_segments,
    extract_single_segment, create_single_spectrogram, create_single_spectrogram_npy, save_test_audio,
    plot_summary, get_spect_matrix, get_spec_matrix_direct, audio_process
)

from utils.training_utils import (
    train_epoch, validate_epoch, train_single_fold, 
    fast_single_fold_training_with_predefined_split,
    fast_single_fold_training_with_augmentation,
    single_fold_training_with_predefined_split
)

from utils.evaluation_utils import (
    get_confusion_matrix, plot_confusion_matrix, plot_best_results,
    plot_mean_curve, plot_kfold_results, plot_single_fold_curve,
    print_single_fold_results, print_confusion_matrix_stats,
    print_kfold_best_results, save_model, test_saved_model,
    load_model, reset_model
)

from utils.cross_validation import (
    k_fold_cross_validation, k_fold_cross_validation_with_predefined_folds
)

from utils.dataset_utils import create_augmented_dataset_wrapper

# Legacy compatibility notice
def __getattr__(name):
    """Provide helpful error messages for moved functions."""
    moved_functions = {
        # Data processing functions
        'clean_dir': 'utils.data_processing',
        'count_files_in_dir': 'utils.data_processing', 
        'lbrs_loading': 'utils.data_processing',
        'get_rmsThreshold': 'utils.data_processing',
        'reduce_noise_seg': 'utils.data_processing',
        'get_spec_norm': 'utils.data_processing',
        'get_spec_image': 'utils.data_processing',
        'save_test_audios': 'utils.data_processing',
        'audio_process': 'utils.data_processing',
        
        # Training functions  
        'train_epoch': 'utils.training_utils',
        'validate_epoch': 'utils.training_utils',
        'train_single_fold': 'utils.training_utils',
        
        # Evaluation functions
        'get_confusion_matrix': 'utils.evaluation_utils',
        'plot_confusion_matrix': 'utils.evaluation_utils',
        'plot_kfold_results': 'utils.evaluation_utils',
        'save_model': 'utils.evaluation_utils',
        'load_model': 'utils.evaluation_utils',
        
        # Cross-validation functions
        'k_fold_cross_validation': 'utils.cross_validation',
        'k_fold_cross_validation_with_predefined_folds': 'utils.cross_validation',
        
        # Dataset functions
        'StandardizedDataset': 'utils.dataset_utils',
        'StandardizedSubset': 'utils.dataset_utils',
        'FastStandardizedSubset': 'utils.dataset_utils',
    }
    
    if name in moved_functions:
        module = moved_functions[name]
        raise AttributeError(
            f"Function '{name}' has been moved to '{module}'. "
            f"Please import from the new location: from {module} import {name}"
        )
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

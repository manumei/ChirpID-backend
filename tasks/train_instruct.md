# Training Instructions - ChirpID Model Training

## Overview

The ChirpID training system has been refactored into a clean, top-down architecture with two main training functions:

- **`cross_val_training()`** - For K-fold cross-validation training
- **`single_fold_training()`** - For single train/validation split training

## Quick Start Guide

### Step 1: Import Data

Choose one of these methods:

**Method A: Load from CSV file**
```python
from utils.training_core import cross_val_training, single_fold_training
from utils.models import BirdCNN

# Data will be loaded automatically from CSV
data_path = '../database/meta/final/train_data.csv'
```

**Method B: Use existing arrays**
```python
# If you already have features, labels, and authors as arrays
features = your_features_array  # Shape: (N, 1, 313, 224)
labels = your_labels_array      # Shape: (N,)
authors = your_authors_array    # Shape: (N,)
```

### Step 2: Choose Training Type

**For K-Fold Cross-Validation:**
```python
results, best_results = cross_val_training(
    data_path=data_path,  # Or pass features, labels, authors
    model_class=BirdCNN,
    num_classes=31,
    config={
        'k_folds': 4,
        'num_epochs': 220,
        'batch_size': 24,
        'learning_rate': 0.001,
        'use_class_weights': False,
        'early_stopping': 35,
        'standardize': True
    }
)
```

**For Single Fold Training:**
```python
results = single_fold_training(
    data_path=data_path,  # Or pass features, labels, authors
    model_class=BirdCNN,
    num_classes=31,
    config={
        'num_epochs': 250,
        'batch_size': 48,
        'learning_rate': 0.001,
        'use_class_weights': False,
        'early_stopping': 35,
        'standardize': True,
        'test_size': 0.2
    },
    use_predefined_split=True  # True = author-grouped split, False = stratified split
)
```

### Step 3: Visualize Results

**For Cross-Validation:**
```python
from utils.evaluation_utils import plot_kfold_results
plot_kfold_results(results, best_results)
```

**For Single Fold:**
```python
from utils.evaluation_utils import plot_single_fold_curve, print_single_fold_results

plot_single_fold_curve(results, metric_key='accuracies', title="Accuracy Curves", ylabel="Accuracy")
plot_single_fold_curve(results, metric_key='losses', title="Loss Curves", ylabel="Loss")
plot_single_fold_curve(results, metric_key='f1s', title="F1 Score Curves", ylabel="F1 Score")

print_single_fold_results(results)
```

## Configuration Options

### Common Parameters
- `num_epochs`: Training epochs (default: 250 for single, 220 for CV)
- `batch_size`: Batch size (default: 48 for single, 24 for CV)
- `learning_rate`: Learning rate (default: 0.001)
- `use_class_weights`: Enable balanced class weights (default: False)
- `early_stopping`: Early stopping patience (default: 35)
- `standardize`: Apply feature standardization (default: True)

### Cross-Validation Specific
- `k_folds`: Number of folds (default: 4)
- `aggregate_predictions`: Compute aggregated metrics (default: True)
- `max_split_attempts`: Max attempts for optimal split (default: 30000)
- `min_val_segments`: Min segments per class in validation (default: 0)

### Single Fold Specific
- `test_size`: Validation set fraction (default: 0.2)
- `random_state`: Random seed for stratified split (default: 435)
- `max_split_attempts`: Max attempts for optimal split (default: 10000)
- `min_test_segments`: Min segments per class in test set (default: 5)

## Complete Example

```python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.training_core import cross_val_training, single_fold_training
from utils.models import BirdCNN
from utils.evaluation_utils import plot_kfold_results, plot_single_fold_curve, print_single_fold_results

# Load data
data_path = '../database/meta/final/train_data.csv'

# Cross-validation training
print("Starting cross-validation training...")
cv_results, cv_best = cross_val_training(
    data_path=data_path,
    model_class=BirdCNN,
    num_classes=31,
    config={
        'k_folds': 4,
        'num_epochs': 200,
        'batch_size': 24,
        'learning_rate': 0.001,
        'standardize': True
    }
)

# Plot CV results
plot_kfold_results(cv_results, cv_best)

# Single fold training
print("Starting single fold training...")
single_results = single_fold_training(
    data_path=data_path,
    model_class=BirdCNN,
    num_classes=31,
    config={
        'num_epochs': 250,
        'batch_size': 48,
        'learning_rate': 0.001,
        'standardize': True
    },
    use_predefined_split=True
)

# Plot single fold results
plot_single_fold_curve(single_results, 'accuracies', "Accuracy", "Accuracy")
plot_single_fold_curve(single_results, 'f1s', "F1 Score", "F1 Score")
print_single_fold_results(single_results)

print("Training completed!")
```

## Architecture Notes

The new system uses a modular design:

- **`training_core.py`** - Top-level training functions (entry points)
- **`training_engine.py`** - Core training execution engine
- **`data_preparation.py`** - Data loading and preprocessing
- **`split.py`** - Dataset splitting with author grouping
- **Legacy files** - Maintained for backward compatibility

The system automatically handles:
- Data loading and preprocessing
- Optimal train/validation splits with author grouping
- Feature standardization
- Class weight computation
- Early stopping
- Result aggregation and visualization

This provides a clean, easy-to-use interface while maintaining all the advanced functionality of the original system.

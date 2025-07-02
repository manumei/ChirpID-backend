# Utils Directory Documentation

This document describes the purpose and contents of each file in the `utils/` directory after the modularization refactoring.

## File Overview

### Core Utility Modules (New Modular Structure)

#### `data_processing.py`
**Purpose**: Audio processing, file I/O, and spectrogram creation
**Functions**:
- `clean_dir()` - Directory cleanup utilities
- `count_files_in_dir()` - File counting utilities
- `lbrs_loading()` - Librosa audio loading
- `get_rmsThreshold()` - RMS threshold calculation
- `reduce_noise_seg()` - Noise reduction for audio segments
- `get_spec_norm()` - Normalized spectrogram generation
- `get_spec_image()` - Spectrogram image creation
- `save_audio_segments_to_disk()` - Audio segment persistence
- `load_audio_segments_from_disk()` - Audio segment loading
- `load_audio_files()` - Bulk audio file loading
- `extract_balanced_segments()` - Balanced audio segment extraction
- `create_single_spectrogram()` - Individual spectrogram creation
- `plot_summary()` - Data summary visualization
- `get_spect_matrix()` - Spectrogram matrix extraction
- `audio_process()` - Main audio processing pipeline

#### `dataset_utils.py`
**Purpose**: PyTorch dataset classes, data standardization, and augmentation support
**Classes**:
- `StandardizedDataset` - Custom dataset for standardized data with multiprocessing support
- `StandardizedSubset` - Subset with automatic standardization for pickling compatibility
- `FastStandardizedSubset` - Optimized subset with tensor indices for faster training
**Functions**:
- `get_spect_matrix_list()` - Load spectrograms directly as matrices without CSV flattening
- `load_spectrograms_from_csv()` - Load spectrogram matrices from CSV files

#### `training_utils.py`
**Purpose**: Model training, validation, and optimization
**Classes**:
- `EarlyStopping` - Early stopping implementation with patience and model saving
**Functions**:
- `train_single_fold()` - Single fold training with comprehensive metrics tracking
- `fast_single_fold_training_with_augmentation()` - Optimized training with SpecAugment support
- Training utility functions for model optimization and performance tracking

#### `cross_validation.py`
**Purpose**: K-fold cross-validation implementation
**Functions**:
- `k_fold_cross_validation_with_predefined_folds()` - K-fold training with author-grouped splits
- Cross-validation logic with proper author grouping to prevent data leakage
- Fold-wise training and aggregation of results

#### `metrics.py`
**Purpose**: Model evaluation, visualization, and result analysis
**Functions**:
- `plot_kfold_results()` - K-fold results visualization with accuracy and loss curves
- `plot_single_fold_curve()` - Individual training curve plotting
- `print_single_fold_results()` - Results summary printing
- `plot_confusion_matrix()` - Confusion matrix visualization
- `print_confusion_matrix_stats()` - Confusion matrix statistics
- `save_model()` - Model persistence utilities
- `load_model()` - Model loading utilities
- Evaluation and visualization functions for model performance analysis

### Supporting Modules (Existing)

#### `models.py`
**Purpose**: Neural network model definitions
**Classes**:
- `BirdCNN` - Convolutional neural network for bird species classification
  - Gradual channel progression (32→64→128→256)
  - Global average pooling to reduce parameters
  - Batch normalization and dropout for regularization
  - Configurable dropout rates
- `BirdResNet` - Residual network architecture for bird classification
  - ResNet-style residual blocks
  - Adaptive global pooling
  - Batch normalization and skip connections
- `ResidualBlock` - Building block for ResNet architecture

#### `split.py`
**Purpose**: Data splitting with author grouping to prevent data leakage
**Functions**:
- `try_split_with_seed()` - Attempt stratified split with given seed
- `search_best_group_seed()` - Find optimal train/test split maintaining author groups
- `try_kfold_split_with_seed()` - K-fold splitting with author grouping
- `search_best_group_seed_kfold()` - Find optimal K-fold configuration
- Author-grouped splitting ensures no author appears in both train and test sets

#### `specaugment.py`
**Purpose**: SpecAugment data augmentation for spectrograms
**Classes**:
- `SpecAugment` - Time and frequency masking for spectrograms
- `PILSpecAugment` - PIL-based augmentation implementation
**Functions**:
- `get_recommended_params()` - Calculate optimal augmentation parameters based on dataset
- `visualize_specaugment()` - Visualization of augmentation effects
- Dynamic parameter recommendation based on dataset size and characteristics

#### `NnClass.py`
**Purpose**: Legacy neural network utilities (consider deprecating)
**Note**: Contains older neural network implementations that may be redundant with models.py

#### `util.py` (Refactored)
**Purpose**: Legacy utilities and audio processing functions (partially cleaned)
**Note**: Now imports from new modular structure. Contains remaining audio processing functions that weren't moved to data_processing.py. This file should be further cleaned in future refactoring.

## Migration Guide

### Old vs New Imports
```python
# OLD - Monolithic imports
from utils import util, models, split
util.function_name()

# NEW - Modular imports  
from utils.training_utils import train_single_fold
from utils.metrics import plot_confusion_matrix
from utils.dataset_utils import StandardizedDataset
from utils.models import BirdCNN
```

### Key Benefits of New Structure
1. **Modularity**: Related functions grouped logically
2. **Maintainability**: Easier to locate and modify specific functionality
3. **Testability**: Each module can be tested independently
4. **Clarity**: Clear separation of concerns
5. **Reusability**: Modules can be imported independently
6. **Scalability**: Easy to extend without affecting other components

### Future Improvements
1. Complete migration of remaining util.py functions to appropriate modules
2. Deprecate and remove NnClass.py if redundant
3. Add comprehensive unit tests for each module
4. Consider splitting large modules further if they grow
5. Add type hints throughout the codebase

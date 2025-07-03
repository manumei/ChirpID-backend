# Utils Directory

This directory contains utility modules for the ChirpID bird species classification system. Each module provides specialized functionality for different aspects of the machine learning pipeline, from data processing to model training and evaluation.

## Core Files

### `data_preparation.py`

Handles data loading and preprocessing for training. Contains functions to:

- Load training data from CSV files or numpy arrays
- Prepare features, labels, and author information
- Normalize features to 0-1 range
- Reshape data for CNN input (batch_size, channels, height, width)
- Validate data consistency and format

### `data_processing.py`

Comprehensive audio processing utilities for the entire data pipeline. Includes:

- **Directory utilities**: Clean directories, count files
- **Audio loading**: Load audio files with librosa and sample rate validation
- **Audio segmentation**: RMS threshold calculation, noise reduction
- **Spectrogram generation**: Create normalized mel spectrograms
- **File I/O**: Save spectrograms as images (.png) or numpy arrays (.npy)
- **Batch processing**: Handle multiple audio files with progress tracking

### `models.py`

Contains PyTorch Convolutional Neural Network architectures for bird species classification:

- **ResidualBlock**: Basic building block with skip connections for deeper networks
- **Multiple CNN architectures**: Various convolutional neural network designs
- **Attention mechanisms**: Self-attention and channel attention modules
- **Custom layers**: Specialized layers for spectrogram processing
- **Model utilities**: Helper functions for model initialization and configuration

### `fcnn_models.py`

Contains by-hand and PyTorch implementations of Fully-Connected Neural Networks for bird species classification.
Similar to models.py but for flattened vectors in non-convolutional networks.

### `training_core.py`

Top-level training interface with clean, hierarchical structure:

- **Cross-validation training**: K-fold cross-validation with author grouping
- **Single-fold training**: Train-validation split functionality
- **Pre-computed splits optimization**: Reuse splits across hyperparameter sweeps
- **Training orchestration**: High-level coordination of training processes
- **Configuration management**: Handle training parameters and settings

### `training_engine.py`

Core training execution logic and low-level training operations:

- **Training loops**: Actual epoch-by-epoch training execution
- **Data loading management**: Optimized DataLoader configuration
- **Model lifecycle**: Initialization, training, validation, and checkpointing
- **Optimization**: Learning rate scheduling, early stopping, gradient clipping
- **Metrics tracking**: Loss calculation, accuracy monitoring, F1 scores
- **Memory management**: GPU memory optimization and garbage collection

## Data Handling

### `dataset_utils.py`

PyTorch dataset creation and data standardization utilities:

- **Custom Dataset classes**: StandardizedDataset, StandardizedSubset, FastStandardizedSubset
- **Data standardization**: Compute and apply training statistics (mean/std)
- **Augmentation integration**: Wrapper classes for data augmentation
- **Worker-safe implementations**: Multiprocessing-compatible dataset classes
- **Memory optimization**: Efficient data loading for large datasets

### `dataloader_factory.py`

Hardware-optimized DataLoader configuration factory:

- **Optimal configurations**: Hardware-specific settings for RTX 5080 + Ryzen 9 7950X
- **Worker management**: Automatic worker count optimization based on operations
- **Memory settings**: Pin memory, prefetch factor, persistent workers
- **Batch size optimization**: Adaptive batch sizing based on dataset characteristics
- **Performance tuning**: Maximize training throughput while maintaining stability

### `split.py`

Data splitting functions for train/validation/test sets:

- **Author-grouped splits**: Prevent data leakage by grouping by audio authors
- **Quality scoring**: Evaluate split quality based on class distribution
- **Seed search**: Find optimal random seeds for balanced splits
- **Cross-validation support**: K-fold splitting with group constraints
- **Validation**: Ensure all classes present in all splits with minimum thresholds

## Model Support

### `inference.py`

Model inference and prediction utilities:

- **Model loading**: Load pretrained CNN models with weights
- **Tensor conversion**: Convert audio segments to model-ready tensors
- **Batch prediction**: Efficient inference on multiple samples
- **Post-processing**: Convert model outputs to class predictions
- **Device management**: Handle CPU/GPU inference automatically

### `metrics.py`

Evaluation and visualization utilities for model performance:

- **Confusion matrices**: Generate and visualize confusion matrices
- **Performance metrics**: Calculate accuracy, F1 scores, precision, recall
- **Visualization**: Plot training curves, validation metrics
- **Result analysis**: Detailed performance breakdowns per class
- **Comparative analysis**: Compare multiple model performances

### `specaugment.py`

SpecAugment data augmentation for spectrograms:

- **Time masking**: Mask consecutive time steps (vertical bands)
- **Frequency masking**: Mask consecutive frequency bins (horizontal bands)
- **Gaussian noise**: Add random noise to spectrograms
- **PIL integration**: PIL-based augmentation for image-format spectrograms
- **Parameter optimization**: Recommended parameters for different spectrogram sizes
- **Training-only application**: Automatic train/eval mode handling

## Legacy and Deprecated

### `oldmodels.py`

**DEPRECATED** - Legacy model architectures that are no longer maintained:

- Contains old CNN implementations (OldBirdCNN, BirdCNN)
- **Do not use** - kept only for backwards compatibility
- Use `models.py` for current model architectures

## Usage Examples

```python
# Data preparation
from utils.data_preparation import prepare_training_data
features, labels, authors = prepare_training_data('data/train.csv')

# Training
from utils.training_core import cross_val_training
from utils.models import ResNetBirdCNN
results = cross_val_training(
    features=features, labels=labels, authors=authors,
    model_class=ResNetBirdCNN, num_classes=33
)

# Inference
from utils.inference import load_model_weights
model, device = load_model_weights(ResNetBirdCNN, 'models/best_model.pth')
```

## Dependencies

This utils package requires:

- PyTorch and torchvision
- NumPy and pandas
- scikit-learn
- librosa and soundfile
- matplotlib and seaborn
- PIL (Pillow)
- tqdm for progress bars
- noisereduce for audio preprocessing (tho kinda deprecated tbh)

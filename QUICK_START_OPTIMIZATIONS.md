# ChirpID Optimizations Quick Start Guide

## 🚀 Quick Setup for RTX 5080 + Ryzen 9 7950X

### Step 1: Basic Optimized Training
```python
from utils.training_core import cross_val_training

# Basic configuration with all optimizations enabled
config = {
    'mixed_precision': True,        # 40-60% speed boost
    'gradient_clipping': 1.0,       # Training stability
    'batch_size': 32,               # Larger batches with AMP
    'num_epochs': 220,
    'k_folds': 4
}

results = cross_val_training(
    data_path="path/to/your/data.csv",
    model_class=YourModelClass,
    num_classes=10,
    config=config
)
```

### Step 2: Maximum Performance (Parallel Folds)
```python
# Maximum performance configuration
config = {
    'mixed_precision': True,        # Essential for RTX 5080
    'gradient_clipping': 1.0,       # Training stability
    'parallel_folds': True,         # 2-3x faster cross-validation
    'max_parallel_folds': 2,        # Conservative for 16GB VRAM
    'batch_size': 32,               # Optimized for AMP
    'num_epochs': 220,
    'k_folds': 4,
    'standardize': True,
    'spec_augment': True            # Enhanced data augmentation
}

results = cross_val_training(
    data_path="path/to/your/data.csv",
    model_class=YourModelClass,
    num_classes=10,
    config=config
)
```

### Step 3: Debug and Monitor
```python
# Configuration with debugging enabled
config = {
    'mixed_precision': True,
    'parallel_folds': True,
    'max_parallel_folds': 2,
    'debug_dataloaders': True,      # See DataLoader configuration
    'benchmark_performance': True,  # Measure improvements
}
```

## 🔧 Key Optimizations Enabled

- ✅ **Mixed Precision Training**: ~50% faster, less memory
- ✅ **Parallel Fold Training**: 2-3x faster cross-validation
- ✅ **Optimized DataLoaders**: 12 workers, pin memory, persistent workers
- ✅ **Gradient Clipping**: Improved training stability
- ✅ **Memory Management**: Automatic figure cleanup

## 📊 Expected Performance Gains

| Optimization | Speed Improvement | Memory Reduction |
|-------------|-------------------|------------------|
| Mixed Precision | 40-60% | 20-30% |
| Parallel Folds | 200-300% | - |
| DataLoader Opts | 20-30% | - |
| **Combined** | **~400-500%** | **20-30%** |

## ⚠️ Important Notes

1. **GPU Memory**: Start with `max_parallel_folds=2` for RTX 5080
2. **CPU Usage**: Will use ~12 CPU cores during training
3. **Compatibility**: All optimizations are backward compatible
4. **Monitoring**: Check GPU memory usage during first runs

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors
```python
# Reduce parallel folds or batch size
config = {
    'max_parallel_folds': 1,  # Sequential fallback
    'batch_size': 16,         # Smaller batches
}
```

### Slow DataLoader Performance  
```python
# Debug DataLoader configuration
config = {
    'debug_dataloaders': True,
    'optimize_dataloaders': True
}
```

### Check Current Configuration
```python
# All configurations are logged at training start
# Look for output like:
# "Using PARALLEL fold training (max 2 concurrent folds)"
# "Mixed precision training: ENABLED"
```

## 🎯 Recommended Usage Patterns

### Development/Testing
```python
config = {'mixed_precision': True}  # Quick and safe
```

### Production/Research
```python
config = {
    'mixed_precision': True,
    'parallel_folds': True,
    'max_parallel_folds': 2
}
```

### Maximum Performance (Advanced)
```python
config = {
    'mixed_precision': True,
    'parallel_folds': True,
    'max_parallel_folds': 3,  # Aggressive - monitor GPU memory
    'batch_size': 40,         # Larger batches
}
```

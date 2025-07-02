# Model Configuration Guide

This guide explains how to properly configure hyperparameters for bird song classification using the ModelConfiguring notebook.

## Dataset Context
- **Samples**: ~3,200 training samples, ~800 validation samples
- **Classes**: ~31 bird species
- **Input**: 313×224 grayscale log-mel spectrograms
- **Task**: Multi-class audio classification

## Hyperparameter Reference

### 1. Optimizer Selection (`use_adam`)
**Type**: Boolean or String  
**Default**: `True`
**Options**: `False`, `True`/`'adam'`, `'adamw'`

**Description**: Chooses between SGD, Adam, and AdamW optimizers.

**Guidelines**:
- **SGD (`False`)**: Traditional optimizer with momentum, can achieve best final performance but requires careful tuning
- **Adam (`True` or `'adam'`)**: Adaptive learning rate optimizer, generally better for deep learning, works well with default settings
- **AdamW (`'adamw'`)**: Improved version of Adam with better weight decay handling, recommended for large models

**For our dataset**:
- **AdamW** is recommended for most experiments (best generalization)
- **Adam** is good for quick prototyping and baseline experiments
- **SGD** requires 5-10x higher learning rates (0.01-0.05 vs 0.001-0.005) but can achieve superior final performance
- With ~3,200 samples, Adam's adaptive nature helps with small dataset challenges

### 2. Early Stopping Threshold (`estop_thresh`)
**Type**: Integer  
**Default**: `35`

**Description**: Number of epochs without validation improvement before stopping training.

**Guidelines**:
- **Conservative (40-80)**: Allows more exploration, good for complex models
- **Moderate (25-40)**: Balanced approach, prevents severe overfitting
- **Aggressive (10-25)**: Fast training, risk of premature stopping

**For our dataset**:
- With 31 classes and limited data, use 30-50 epochs patience
- Smaller datasets need more patience due to validation noise
- SpecAugment training may need higher thresholds (40-60)

### 3. Batch Size (`batch_size`)
**Type**: Integer  
**Default**: `24`

**Description**: Number of samples processed simultaneously.

**Guidelines**:
- **Small (16-32)**: Better gradient estimates, more frequent updates, higher memory efficiency
- **Medium (32-64)**: Good balance of speed and stability
- **Large (64+)**: Faster training, may need higher learning rates, less noisy gradients

**For our dataset**:
- With ~3,200 samples: 16-48 batch size recommended
- 24-32 provides good balance for our dataset size
- Avoid very large batches (>64) with limited data

**Dependencies**: 
- Larger batches may require higher learning rates
- Memory constraints may limit maximum batch size

### 4. Class Weights (`use_class_weights`)
**Type**: Boolean  
**Default**: `False`

**Description**: Automatically weights classes inversely proportional to their frequency.

**Guidelines**:
- **Enable**: When classes are significantly imbalanced (>3:1 ratio)
- **Disable**: When classes are relatively balanced or when using other balancing techniques

**For our dataset**:
- Check class distribution first
- If some species have <50 samples while others have >150, enable class weights
- Monitor for potential over-correction (too much focus on rare classes)

### 5. L2 Regularization (`l2_regularization`)
**Type**: Float  
**Default**: `1e-4`

**Description**: Weight decay parameter to prevent overfitting.

**Guidelines**:
- **Light (1e-5 to 1e-4)**: For well-regularized models or large datasets
- **Moderate (1e-4 to 5e-4)**: Standard range for most applications
- **Heavy (1e-3+)**: For overfitting issues, but may hurt performance

**For our dataset**:
- Start with 1e-4 to 3e-4 range
- Increase if validation loss plateaus while training loss decreases
- With limited data, moderate regularization (2e-4 to 5e-4) often optimal

**Dependencies**: 
- Higher L2 may require lower learning rates
- Interacts with early stopping (may need longer patience)

### 6. Learning Rate Schedule (`lr_schedule`)
**Type**: Dictionary or None  
**Default**: `None`

**Description**: How learning rate changes during training.

**Options**:
```python
# No scheduling
lr_schedule = None

# Exponential decay
lr_schedule = {'type': 'exponential', 'gamma': 0.95}

# Reduce on plateau
lr_schedule = {'type': 'plateau', 'factor': 0.5, 'patience': 10}

# Cosine annealing
lr_schedule = {'type': 'cosine', 'T_max': 50}
```

**Guidelines**:
- **No schedule**: Good for initial experiments, Adam handles adaptation
- **Exponential**: Gradual decay, good for long training
- **Plateau**: Reactive to validation performance, conservative
- **Cosine**: Aggressive decay with restarts, good for fine-tuning

**For our dataset**:
- Start without scheduling for baseline
- Plateau scheduling works well with early stopping
- Exponential decay (γ=0.95-0.98) for longer training runs

**Dependencies**: 
- **Critical**: Must consider initial learning rate
- Higher initial LR may need more aggressive scheduling
- Early stopping patience should accommodate schedule

### 7. Initial Learning Rate (`initial_lr`)
**Type**: Float  
**Default**: `0.001`

**Description**: Starting learning rate for optimization.

**Guidelines**:
- **Adam**: 0.0005-0.003 typical range
- **SGD**: 0.01-0.1 typical range
- **With scheduling**: Can start higher (2-5x)
- **Without scheduling**: More conservative

**For our dataset**:
- Adam: 0.001-0.002 recommended starting point
- SGD: 0.01-0.03 recommended starting point
- Limited data suggests moderate learning rates

**Dependencies**: 
- **Critical with batch size**: Larger batches may need higher LR
- **Critical with optimizer**: SGD needs 5-10x higher rates than Adam
- **Critical with scheduling**: Scheduled training can start higher

### 8. Standardization (`standardize`)
**Type**: Boolean  
**Default**: `True`

**Description**: Normalizes input features to zero mean and unit variance per channel.

**Guidelines**:
- **Enable**: Almost always beneficial for deep learning
- **Disable**: Only when features are already normalized or for specific architectures

**For our dataset**:
- **Strongly recommended**: Spectrogram values vary widely
- Helps with training stability and convergence
- Essential when combining different preprocessing approaches

### 9. SpecAugment (`spec_augment`)
**Type**: Boolean  
**Default**: `False`

**Description**: Applies time and frequency masking to spectrograms during training.

**Guidelines**:
- **Enable**: For improving generalization, especially with limited data
- **Disable**: If training time is critical or data is very limited

**For our dataset**:
- **Recommended**: With ~3,200 samples, augmentation helps generalization
- Particularly effective for audio/spectrogram tasks
- May require longer training (higher early stopping threshold)

**Dependencies**: 
- May need increased early stopping patience (40-60 epochs)
- Can interact with learning rate (sometimes need slightly lower rates)

### 10. Noise Augmentation (`noise_augment`)
**Type**: Boolean  
**Default**: `False`

**Description**: Adds Gaussian noise to inputs during training.

**Guidelines**:
- **Enable**: For robustness to recording noise, limited data scenarios
- **Disable**: When data already contains sufficient natural noise variation

**For our dataset**:
- **Moderately recommended**: Bird recordings have natural noise variation
- Less critical than SpecAugment for spectrograms
- Combines well with SpecAugment for comprehensive augmentation

## Parameter Interactions

### Critical Dependencies

1. **Learning Rate ↔ Optimizer**
   - SGD needs 5-10x higher learning rates than Adam
   - Never use Adam learning rates with SGD

2. **Learning Rate ↔ Batch Size**
   - Larger batches often need proportionally higher learning rates
   - Linear scaling rule: double batch size → double learning rate (approximately)

3. **Learning Rate ↔ LR Schedule**
   - With scheduling, can start 2-5x higher
   - Schedule aggressiveness must match initial rate

4. **Early Stopping ↔ Augmentation**
   - Augmentation makes training more noisy
   - Increase patience by 15-30 epochs when using augmentation

5. **L2 Regularization ↔ Learning Rate**
   - Higher L2 may require slightly lower learning rates
   - Both control model complexity differently

### Recommended Starting Combinations

**Conservative Baseline**:
```python
config = {
    'use_adam': True,
    'initial_lr': 0.001,
    'batch_size': 24,
    'estop_thresh': 35,
    'l2_regularization': 1e-4,
    'lr_schedule': None,
    'use_class_weights': False,
    'standardize': True,
    'spec_augment': False,
    'noise_augment': False
}
```

**Augmented Training**:
```python
config = {
    'use_adam': True,
    'initial_lr': 0.0015,
    'batch_size': 24,
    'estop_thresh': 45,  # Higher patience for augmentation
    'l2_regularization': 2e-4,
    'lr_schedule': {'type': 'plateau', 'factor': 0.7, 'patience': 12},
    'use_class_weights': True,
    'standardize': True,
    'spec_augment': True,
    'noise_augment': True
}
```

**Fast Convergence**:
```python
config = {
    'use_adam': True,
    'initial_lr': 0.003,
    'batch_size': 48,
    'estop_thresh': 20,
    'l2_regularization': 1e-4,
    'lr_schedule': None,
    'use_class_weights': False,
    'standardize': True,
    'spec_augment': False,
    'noise_augment': False
}
```

## Expected Performance Ranges

### For Our Dataset (31 classes, ~3,200 samples)

**Typical Results**:
- **Validation Accuracy**: 0.65-0.85
- **Validation F1 Score**: 0.60-0.80
- **Training Time**: 5-25 minutes per configuration

**Performance Indicators**:
- **Good**: F1 > 0.70, Accuracy > 0.75
- **Excellent**: F1 > 0.75, Accuracy > 0.80
- **Outstanding**: F1 > 0.80, Accuracy > 0.85

**Red Flags**:
- Training accuracy >> Validation accuracy (overfitting)
- Both accuracies plateau early (<0.60) (underfitting)
- Training loss increases after initial decrease (learning rate too high)
- No improvement after 50+ epochs (learning rate too low or poor configuration)

## Optimization Strategy

1. **Start with baseline**: Conservative configuration first
2. **Add augmentation**: If baseline works, try augmented version
3. **Tune learning rate**: If convergence is slow/fast, adjust LR
4. **Optimize batch size**: Try larger batches with proportional LR increase
5. **Add scheduling**: For final performance boost
6. **Balance regularization**: Adjust L2 and early stopping based on overfitting signs

## Common Mistakes to Avoid

1. **Using SGD learning rates with Adam** (too high)
2. **Not adjusting early stopping with augmentation** (stopping too early)
3. **Ignoring class imbalance** (when ratio > 3:1)
4. **Over-regularizing small datasets** (L2 > 1e-3)
5. **Batch size too large for dataset size** (>64 for 3,200 samples)
6. **Learning rate scheduling without patience consideration**

Remember: The ModelConfiguring notebook tests 20 different combinations systematically. Use the results to understand parameter interactions specific to your dataset!

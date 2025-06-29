# Training and Validation Utilities
# Handles model training, validation, cross-validation, and evaluation metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import os

from utils.dataset_utils import (
    StandardizedDataset, StandardizedSubset, FastStandardizedSubset,
    compute_standardization_stats, create_standardized_subset,
    create_augmented_dataset_wrapper
)
from utils.evaluation_utils import get_confusion_matrix
from utils.dataloader_factory import OptimalDataLoaderFactory

# Core Training Functions
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch and return loss, accuracy, and F1 score."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y_batch.detach().cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return running_loss / total, correct / total, f1

def validate_epoch(model, val_loader, criterion, device, return_predictions=False):
    """Validate model for one epoch and return loss, accuracy, and F1 score."""
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_targets = [], []
    all_predictions, all_target_tensors = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(y_batch.detach().cpu().numpy())
            
            if return_predictions:
                all_predictions.append(outputs.detach().cpu())
                all_target_tensors.append(y_batch.detach().cpu())
    
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    if return_predictions:
        return val_loss / val_total, val_correct / val_total, f1, torch.cat(all_predictions), torch.cat(all_target_tensors)
    else:
        return val_loss / val_total, val_correct / val_total, f1

def train_single_fold(model, train_loader, val_loader, criterion, optimizer, 
                     num_epochs, device, fold_num=None, estop=35, scheduler=None):
    """Train model on a single fold and return training history including F1 scores."""
    
    history = {
        'train_losses': [], 'train_accuracies': [], 'train_f1s': [],
        'val_losses': [], 'val_accuracies': [], 'val_f1s': [],
        'learning_rates': [], 'early_stopped': False, 'best_epoch': 0, 'total_epochs': 0
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    fold_desc = f"Fold {fold_num}" if fold_num else "Training"
    
    for epoch in tqdm(range(num_epochs), desc=f"{fold_desc}"):
        # Training phase
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        # Record metrics
        history['train_losses'].append(train_loss)
        history['train_accuracies'].append(train_acc)
        history['train_f1s'].append(train_f1)
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_acc)
        history['val_f1s'].append(val_f1)
        
        # Learning rate scheduling
        if scheduler:
            # Handle different scheduler types
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                # For other schedulers (ExponentialLR, CosineAnnealingLR), step without metric
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history['best_epoch'] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= estop:
            print(f"Early stopping at epoch {epoch + 1}")
            history['early_stopped'] = True
            break
        
        history['total_epochs'] = epoch + 1
        
        # Progress update every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f} - "
                  f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
    
    return history

# Legacy Support Functions
# These functions are maintained for backward compatibility but users should
# prefer the new training_core.single_fold_training() and training_core.cross_val_training()

def single_fold_training_with_predefined_split(dataset, train_indices, val_indices, model_class, num_classes, 
                                             num_epochs=250, batch_size=48, lr=0.001, 
                                             use_class_weights=True, estop=35, standardize=False):
    """
    Legacy function - use training_core.single_fold_training() instead.
    Perform single fold training with predefined train/validation indices.
    """
    print("⚠️  Warning: This function is deprecated. Use training_core.single_fold_training() instead.")
    
    # Import here to avoid circular imports
    from utils.training_engine import TrainingEngine
    
    config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'use_class_weights': use_class_weights,
        'early_stopping': estop,
        'standardize': standardize
    }
    
    engine = TrainingEngine(model_class, num_classes, config)
    return engine.run_single_fold_predefined(dataset, train_indices, val_indices)

def single_fold_training(dataset, model_class, num_classes, num_epochs=250, batch_size=48, lr=0.001, 
                        test_size=0.2, random_state=435, use_class_weights=True, estop=35):
    """
    Legacy function - use training_core.single_fold_training() instead.
    Perform single fold training with stratified split.
    """
    print("⚠️  Warning: This function is deprecated. Use training_core.single_fold_training() instead.")
    
    # Import here to avoid circular imports
    from utils.training_engine import TrainingEngine
    
    config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'test_size': test_size,
        'random_state': random_state,
        'use_class_weights': use_class_weights,
        'early_stopping': estop,
        'standardize': False
    }
    
    engine = TrainingEngine(model_class, num_classes, config)
    return engine.run_single_fold_stratified(dataset)

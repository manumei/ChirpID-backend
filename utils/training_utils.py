# Training and Validation Utilities
# Handles model training, validation, cross-validation, and evaluation metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
            scheduler.step(val_loss)
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

# Single Fold Training Functions
def single_fold_training_with_predefined_split(dataset, train_indices, val_indices, model_class, num_classes, 
                                             num_epochs=250, batch_size=48, lr=0.001, 
                                             use_class_weights=True, estop=35, standardize=False):
    """
    Perform single fold training with predefined train/validation indices and standardization.
    
    Args:
        dataset: PyTorch dataset containing all data
        train_indices: Indices for training set
        val_indices: Indices for validation set
        model_class: Model class to instantiate
        num_classes: Number of output classes
        num_epochs: Number of epochs to train
        batch_size: Batch size for data loaders
        lr: Learning rate
        use_class_weights: If True, compute and use class weights
        estop: Number of epochs without improvement before early stopping
        standardize: If True, standardize features using training data statistics
    
    Returns:
        Dictionary containing training history, final model, and confusion matrix
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
    
    # Standardization
    if standardize:
        print("Computing standardization statistics...")
        train_mean, train_std = compute_standardization_stats(dataset, train_indices, sample_size=1000)
        print(f"Training data statistics (from sample) - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        
        train_subset = create_standardized_subset(dataset, train_indices, train_mean, train_std)
        val_subset = create_standardized_subset(dataset, val_indices, train_mean, train_std)
        print("Created standardized dataset wrappers")
    else:
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
    
    # Class weights computation
    criterion = nn.CrossEntropyLoss()
    if use_class_weights:
        print("Computing class weights...")
        train_labels = [dataset[i][1].item() for i in train_indices]
        unique_classes = np.unique(train_labels)
        
        class_weights_array = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=train_labels
        )
        
        class_weights = torch.ones(num_classes)
        for i, cls in enumerate(unique_classes):
            class_weights[cls] = class_weights_array[i]
        
        class_weights = class_weights.to(device)
        print(f"Class weights computed: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Data loaders
    print("Creating data loaders...")
    
    # Optimize DataLoader settings based on standardization
    if standardize:
        num_workers = 0  # Custom datasets need single-threaded loading
    else:
        num_workers = min(4, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False, prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False, prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Initialize model and optimizer
    model = model_class(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2, min_lr=1e-6)
    
    # Train the model
    history = train_single_fold(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs, device, fold_num=None, estop=estop, scheduler=scheduler
    )
    
    # Get final validation metrics
    final_val_loss, final_val_acc, final_val_f1 = validate_epoch(model, val_loader, criterion, device)
    
    # Generate confusion matrix
    print("Generating confusion matrix...")

    val_confusion_matrix, val_predictions, val_targets = get_confusion_matrix(model, val_loader, device, num_classes)
    
    results = {
        'history': history,
        'final_val_acc': final_val_acc,
        'final_val_loss': final_val_loss,
        'final_val_f1': final_val_f1,
        'best_val_acc': max(history['val_accuracies']),
        'best_val_f1': max(history['val_f1s']),
        'model': model,
        'model_state': model.state_dict().copy(),
        'confusion_matrix': val_confusion_matrix,
        'val_predictions': val_predictions,
        'val_targets': val_targets,
        'config': {
            'num_epochs': num_epochs, 'batch_size': batch_size, 'learning_rate': lr,
            'device': str(device), 'use_class_weights': use_class_weights,
            'estop': estop, 'standardize': standardize, 'predefined_split': True
        }
    }
    
    if standardize:
        results['train_mean'] = train_mean.item()
        results['train_std'] = train_std.item()
    
    print(f"\nTraining Complete!")
    if history['early_stopped']:
        print(f"Early stopped after {history['total_epochs']} epochs (best at epoch {history['best_epoch'] + 1})")
    print(f"Final - Val Acc: {final_val_acc:.4f}, Val Loss: {final_val_loss:.4f}, Val F1: {final_val_f1:.4f}")
    print(f"Best - Val Acc: {results['best_val_acc']:.4f}, Val F1: {results['best_val_f1']:.4f}")
    
    return results

def fast_single_fold_training_with_predefined_split(dataset, train_indices, val_indices, model_class, num_classes, 
                                                   num_epochs=250, batch_size=48, lr=0.001, 
                                                   use_class_weights=True, estop=35, standardize=False):
    """
    Optimized version of single fold training with minimal startup overhead.
    
    Optimizations:
    - Efficient standardization using sampling
    - Reduced DataLoader workers and overhead
    - Streamlined setup process
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Fast training on {device}")
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
    
    # Quick standardization if needed
    if standardize:
        print("Quick standardization computation...")
        train_mean, train_std = compute_standardization_stats(dataset, train_indices, sample_size=200)
        print(f"Standardization computed from sample - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        
        train_subset = create_standardized_subset(dataset, train_indices, train_mean, train_std, fast=True)
        val_subset = create_standardized_subset(dataset, val_indices, train_mean, train_std, fast=True)
    else:
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
    
    # Quick class weights computation
    criterion = nn.CrossEntropyLoss()
    if use_class_weights:
        print("Computing class weights...")
        # Vectorized approach for class weights
        sample_size = min(len(train_indices), 1000)
        train_labels_tensor = torch.tensor([dataset[i][1].item() for i in train_indices[:sample_size]])
        unique_classes, counts = torch.unique(train_labels_tensor, return_counts=True)
        
        # Simple balanced weighting
        total_samples = len(train_labels_tensor)
        weights = total_samples / (len(unique_classes) * counts.float())
        
        class_weights = torch.ones(num_classes, device=device)
        class_weights[unique_classes] = weights.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
    
    # Optimized DataLoaders (single-threaded to avoid pickling issues)
    print("Creating optimized data loaders...")
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, persistent_workers=False
    )

    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, persistent_workers=False
    )
    
    # Initialize model and training
    print("Initializing model...")
    model = model_class(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2, min_lr=1e-6)
    
    print("Starting training...")
    # Train the model
    history = train_single_fold(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs, device, fold_num=None, estop=estop, scheduler=scheduler
    )
    
    # Get final validation metrics
    final_val_loss, final_val_acc, final_val_f1 = validate_epoch(model, val_loader, criterion, device)
    
    # Generate confusion matrix
    print("Generating confusion matrix...")

    val_confusion_matrix, val_predictions, val_targets = get_confusion_matrix(model, val_loader, device, num_classes)
    
    results = {
        'history': history,
        'final_val_acc': final_val_acc,
        'final_val_loss': final_val_loss,
        'final_val_f1': final_val_f1,
        'best_val_acc': max(history['val_accuracies']),
        'best_val_f1': max(history['val_f1s']),
        'model': model,
        'model_state': model.state_dict().copy(),
        'confusion_matrix': val_confusion_matrix,
        'val_predictions': val_predictions,
        'val_targets': val_targets,
        'config': {
            'num_epochs': num_epochs, 'batch_size': batch_size, 'learning_rate': lr,
            'device': str(device), 'use_class_weights': use_class_weights,
            'estop': estop, 'standardize': standardize, 'predefined_split': True, 'fast_training': True
        }
    }
    
    if standardize:
        results['train_mean'] = train_mean.item()
        results['train_std'] = train_std.item()
    
    print(f"\nTraining Complete!")
    if history['early_stopped']:
        print(f"Early stopped after {history['total_epochs']} epochs (best at epoch {history['best_epoch'] + 1})")
    print(f"Final - Val Acc: {final_val_acc:.4f}, Val Loss: {final_val_loss:.4f}, Val F1: {final_val_f1:.4f}")
    print(f"Best - Val Acc: {results['best_val_acc']:.4f}, Val F1: {results['best_val_f1']:.4f}")
    
    return results

def fast_single_fold_training_with_augmentation(dataset, train_indices, val_indices, model_class, num_classes, 
                                               num_epochs=250, batch_size=48, lr=0.001, 
                                               use_class_weights=True, estop=35, standardize=False,
                                               augment_params=None):
    """
    Optimized single fold training with SpecAugment data augmentation.
    
    Args:
        dataset: PyTorch dataset containing all data
        train_indices: Indices for training set
        val_indices: Indices for validation set
        model_class: Model class to instantiate
        num_classes: Number of output classes
        num_epochs: Number of epochs to train
        batch_size: Batch size for data loaders
        lr: Learning rate
        use_class_weights: If True, compute and use class weights
        estop: Number of epochs without improvement before early stopping
        standardize: If True, standardize features
        augment_params: Dictionary of SpecAugment parameters
    
    Returns:
        Dictionary containing training history and results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training with SpecAugment on {device}")
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
    
    # Step 1: Standardization (same as fast training)
    if standardize:
        print("Quick standardization computation...")
        train_mean, train_std = compute_standardization_stats(dataset, train_indices, sample_size=200)
        print(f"Standardization computed from sample - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        
        base_train_subset = create_standardized_subset(dataset, train_indices, train_mean, train_std, fast=True)
        val_subset = create_standardized_subset(dataset, val_indices, train_mean, train_std, fast=True)
    else:
        base_train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
    
    # Step 2: Add augmentation to training set only
    train_subset = create_augmented_dataset_wrapper(base_train_subset, augment_params, training=True)
    
    # Step 3: Class weights computation (same as before)
    criterion = nn.CrossEntropyLoss()
    if use_class_weights:
        print("Computing class weights...")
        sample_size = min(len(train_indices), 1000)
        train_labels_tensor = torch.tensor([dataset[i][1].item() for i in train_indices[:sample_size]])
        unique_classes, counts = torch.unique(train_labels_tensor, return_counts=True)
        
        total_samples = len(train_labels_tensor)
        weights = total_samples / (len(unique_classes) * counts.float())
        
        class_weights = torch.ones(num_classes, device=device)
        class_weights[unique_classes] = weights.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
    
    # Step 4: DataLoaders
    print("Creating data loaders with augmentation...")
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, persistent_workers=False
    )

    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, persistent_workers=False
    )
    
    # Step 5: Training
    model = model_class(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2, min_lr=1e-6)
    
    history = train_single_fold(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs, device, fold_num=None, estop=estop, scheduler=scheduler
    )
    
    # Final metrics and confusion matrix
    final_val_loss, final_val_acc, final_val_f1 = validate_epoch(model, val_loader, criterion, device)
    
    print("Generating confusion matrix...")

    val_confusion_matrix, val_predictions, val_targets = get_confusion_matrix(model, val_loader, device, num_classes)
    
    results = {
        'history': history,
        'final_val_acc': final_val_acc,
        'final_val_loss': final_val_loss,
        'final_val_f1': final_val_f1,
        'best_val_acc': max(history['val_accuracies']),
        'best_val_f1': max(history['val_f1s']),
        'model': model,
        'model_state': model.state_dict().copy(),
        'confusion_matrix': val_confusion_matrix,
        'val_predictions': val_predictions,
        'val_targets': val_targets,
        'augment_params': augment_params,
        'config': {
            'num_epochs': num_epochs, 'batch_size': batch_size, 'learning_rate': lr,
            'device': str(device), 'use_class_weights': use_class_weights,
            'estop': estop, 'standardize': standardize, 'specaugment': True, 'predefined_split': True
        }
    }
    
    if standardize:
        results['train_mean'] = train_mean.item()
        results['train_std'] = train_std.item()
    
    print(f"\nTraining with SpecAugment Complete!")
    if history['early_stopped']:
        print(f"Early stopped after {history['total_epochs']} epochs (best at epoch {history['best_epoch'] + 1})")
    print(f"Final - Val Acc: {final_val_acc:.4f}, Val Loss: {final_val_loss:.4f}, Val F1: {final_val_f1:.4f}")
    print(f"Best - Val Acc: {results['best_val_acc']:.4f}, Val F1: {results['best_val_f1']:.4f}")
    
    return results

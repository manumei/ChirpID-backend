# Cross-Validation Utilities
# Handles K-fold cross-validation with various optimization strategies

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

def k_fold_cross_validation(dataset, model_class, num_classes, k_folds=4, 
                           num_epochs=300, batch_size=32, lr=0.001, random_state=435, 
                           aggregate_predictions=True, use_class_weights=True, estop=35,
                           standardize=False):
    """
    Perform K-Fold Cross Validation training with F1 score reporting and early stopping.
    Ensures all classes are present in training for each fold.
    
    Args:
        dataset: PyTorch dataset containing all data
        model_class: Model class to instantiate (e.g., models.BirdCNN)
        num_classes: Number of output classes
        k_folds: Number of folds for cross validation
        num_epochs: Number of epochs per fold
        batch_size: Batch size for data loaders
        lr: Learning rate
        random_state: Random seed for reproducibility
        aggregate_predictions: If True, compute cross-entropy on aggregated predictions
        use_class_weights: If True, compute and use class weights for CrossEntropyLoss
        estop: Number of epochs without improvement before early stopping
        standardize: If True, standardize features using training data statistics
    
    Returns:
        Tuple containing:
        - Dictionary containing results for each fold and aggregated metrics including F1 scores
        - Dictionary containing best results for each metric across folds
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract all labels for stratification
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    
    # Check if we have enough samples per class for k-fold CV
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    min_samples_per_class = min(label_counts)
    
    if min_samples_per_class < k_folds:
        print(f"WARNING: Some classes have fewer than {k_folds} samples (minimum: {min_samples_per_class})")
        print("This may cause issues with stratified k-fold CV. Consider reducing k_folds or collecting more data.")
        
    # Try different random states if stratified split fails
    max_attempts = 100
    skfold = None
    
    for attempt in range(max_attempts):
        try:
            current_seed = random_state + attempt
            temp_skfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=current_seed)
            
            # Test if all folds have all classes in training
            all_folds_valid = True
            fold_splits = list(temp_skfold.split(range(len(dataset)), all_labels))
            
            for fold_idx, (train_ids, val_ids) in enumerate(fold_splits):
                train_labels = [all_labels[i] for i in train_ids]
                train_classes = set(train_labels)
                all_classes = set(range(num_classes))
                
                if train_classes != all_classes:
                    missing_classes = all_classes - train_classes
                    print(f"Attempt {attempt + 1}: Fold {fold_idx + 1} missing classes {missing_classes} in training")
                    all_folds_valid = False
                    break
            
            if all_folds_valid:
                skfold = temp_skfold
                final_seed = current_seed
                print(f"Found valid stratified split after {attempt + 1} attempts (seed: {final_seed})")
                break
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
    if skfold is None:
        raise ValueError(f"Could not create valid stratified k-fold splits after {max_attempts} attempts. "
                        "All classes must be present in training for each fold. "
                        "Consider reducing k_folds or ensuring more balanced class distribution.")
    
    # Store results for each fold
    fold_results = {}
    final_val_accuracies = []
    final_val_losses = []
    final_val_f1s = []
    
    # Store best results for each fold
    best_accs = []
    best_f1s = []
    best_losses = []
    
    # For aggregated predictions
    if aggregate_predictions:
        all_final_predictions = []
        all_final_targets = []
    
    print(f"Starting {k_folds}-Fold Stratified Cross Validation on {device}")
    print(f"Dataset size: {len(dataset)}")
    if standardize:
        print("Using standardization based on training data statistics")
    
    # Use the validated fold splits
    for fold, (train_ids, val_ids) in enumerate(fold_splits):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}")
        
        # Verify all classes are present in training (should always pass now)
        train_labels = [all_labels[i] for i in train_ids]
        train_classes = set(train_labels)
        all_classes = set(range(num_classes))
        
        assert train_classes == all_classes, f"Fold {fold + 1} missing classes in training: {all_classes - train_classes}"
        
        print(f"All {num_classes} classes present in training set ✓")
        print(f"{'='*50}")
        
        # Apply standardization if requested
        if standardize:
            train_mean, train_std = compute_standardization_stats(dataset, train_ids)
            print(f"Training data statistics - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
            
            train_subset = create_standardized_subset(dataset, train_ids, train_mean, train_std)
            val_subset = create_standardized_subset(dataset, val_ids, train_mean, train_std)
        else:
            from torch.utils.data import Subset
            train_subset = Subset(dataset, train_ids)
            val_subset = Subset(dataset, val_ids)
        
        # Compute class weights for this fold if enabled
        if use_class_weights:
            # Since we've ensured all classes are present, we can safely compute weights
            all_classes = np.arange(num_classes)
            class_weights_array = compute_class_weight(
                'balanced',
                classes=all_classes,
                y=train_labels
            )
            class_weights = torch.tensor(class_weights_array, dtype=torch.float32).to(device)
            print(f"Class weights computed: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # Create data loaders
        if standardize:
            # For standardized data, we need custom DataLoader handling
            train_dataset = StandardizedDataset(train_subset)
            val_dataset = StandardizedDataset(val_subset)
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=12, pin_memory=True, persistent_workers=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=12, pin_memory=True, persistent_workers=True
            )
        else:
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=12, pin_memory=True, persistent_workers=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                num_workers=12, pin_memory=True, persistent_workers=True
            )
        
        # Initialize model and optimizer
        model = model_class(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2, min_lr=1e-6)
        
        # Train the fold
        fold_history = train_single_fold(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs, device, fold_num=fold+1, estop=estop, scheduler=scheduler
        )
        
        # Get final predictions if aggregating
        if aggregate_predictions:
            final_val_loss, final_val_acc, final_val_f1, final_preds, final_targets = validate_epoch(
                model, val_loader, criterion, device, return_predictions=True
            )
            all_final_predictions.append(final_preds)
            all_final_targets.append(final_targets)
        else:
            final_val_loss = fold_history['val_losses'][-1]
            final_val_acc = fold_history['val_accuracies'][-1]
            final_val_f1 = fold_history['val_f1s'][-1]
        
        # Calculate best values for this fold
        fold_best_acc = max(fold_history['val_accuracies'])
        fold_best_f1 = max(fold_history['val_f1s'])
        fold_best_loss = min(fold_history['val_losses'])
        
        # Store best results
        best_accs.append(fold_best_acc)
        best_f1s.append(fold_best_f1)
        best_losses.append(fold_best_loss)
        
        # Store fold results
        fold_results[f'fold_{fold+1}'] = {
            'history': fold_history,
            'final_val_acc': final_val_acc,
            'final_val_loss': final_val_loss,
            'final_val_f1': final_val_f1,
            'best_val_acc': fold_best_acc,
            'best_val_f1': fold_best_f1,
            'model_state': model.state_dict().copy(),
            'class_weights': class_weights.cpu() if use_class_weights else None
        }
        
        if standardize:
            fold_results[f'fold_{fold+1}']['train_mean'] = train_mean.item()
            fold_results[f'fold_{fold+1}']['train_std'] = train_std.item()
        
        final_val_accuracies.append(final_val_acc)
        final_val_losses.append(final_val_loss)
        final_val_f1s.append(final_val_f1)
    
    # Calculate aggregate statistics
    if aggregate_predictions:
        # Compute true aggregated cross-entropy
        all_predictions = torch.cat(all_final_predictions, dim=0)
        all_targets = torch.cat(all_final_targets, dim=0)
        
        criterion_agg = nn.CrossEntropyLoss()
        aggregated_loss = criterion_agg(all_predictions, all_targets).item()
        
        # Compute aggregated accuracy and F1
        aggregated_preds = all_predictions.argmax(dim=1)
        aggregated_accuracy = (aggregated_preds == all_targets).float().mean().item()
        aggregated_f1 = f1_score(all_targets.numpy(), aggregated_preds.numpy(), average='macro', zero_division=0)
        
        summary = {
            'aggregated_accuracy': aggregated_accuracy,
            'aggregated_loss': aggregated_loss,
            'aggregated_f1': aggregated_f1,
            'mean_val_accuracy': np.mean(final_val_accuracies),
            'std_val_accuracy': np.std(final_val_accuracies),
            'mean_val_loss': np.mean(final_val_losses),
            'std_val_loss': np.std(final_val_losses),
            'mean_val_f1': np.mean(final_val_f1s),
            'std_val_f1': np.std(final_val_f1s),
            'individual_accuracies': final_val_accuracies,
            'individual_losses': final_val_losses,
            'individual_f1s': final_val_f1s
        }
    else:
        # Use mean of fold losses (original approach)
        mean_val_acc = np.mean(final_val_accuracies)
        std_val_acc = np.std(final_val_accuracies)
        mean_val_loss = np.mean(final_val_losses)
        std_val_loss = np.std(final_val_losses)
        mean_val_f1 = np.mean(final_val_f1s)
        std_val_f1 = np.std(final_val_f1s)
        
        summary = {
            'mean_val_accuracy': mean_val_acc,
            'std_val_accuracy': std_val_acc,
            'mean_val_loss': mean_val_loss,
            'std_val_loss': std_val_loss,
            'mean_val_f1': mean_val_f1,
            'std_val_f1': std_val_f1,
            'individual_accuracies': final_val_accuracies,
            'individual_losses': final_val_losses,
            'individual_f1s': final_val_f1s
        }
    
    # Compile results
    results = {
        'fold_results': fold_results,
        'summary': summary,
        'config': {
            'k_folds': k_folds,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'device': str(device),
            'aggregate_predictions': aggregate_predictions,
            'use_class_weights': use_class_weights,
            'standardize': standardize,
            'final_random_state': final_seed
        }
    }
    
    # Create best results dictionary
    best_results = {
        'accuracies': best_accs,
        'f1s': best_f1s,
        'losses': best_losses
    }
    
    return results, best_results

def k_fold_cross_validation_with_predefined_folds(dataset, fold_indices, model_class, num_classes, 
                                                  num_epochs=300, batch_size=32, lr=0.001, 
                                                  aggregate_predictions=True, use_class_weights=True, 
                                                  estop=35, standardize=False):
    """
    Perform K-Fold Cross Validation training with predefined fold indices.
    
    Args:
        dataset: PyTorch dataset containing all data
        fold_indices: List of (train_indices, val_indices) tuples for each fold
        model_class: Model class to instantiate (e.g., models.BirdCNN)
        num_classes: Number of output classes
        num_epochs: Number of epochs per fold
        batch_size: Batch size for data loaders
        lr: Learning rate
        aggregate_predictions: If True, compute cross-entropy on aggregated predictions
        use_class_weights: If True, compute and use class weights for CrossEntropyLoss
        estop: Number of epochs without improvement before early stopping
        standardize: If True, standardize features using training data statistics
    
    Returns:
        Tuple containing:
        - Dictionary containing results for each fold and aggregated metrics
        - Dictionary containing best results for each metric across folds
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    k_folds = len(fold_indices)
    
    # Store results for each fold
    fold_results = {}
    final_val_accuracies = []
    final_val_losses = []
    final_val_f1s = []
    
    # Store best results for each fold
    best_accs = []
    best_f1s = []
    best_losses = []
    
    # For aggregated predictions
    if aggregate_predictions:
        all_final_predictions = []
        all_final_targets = []
    
    print(f"Starting {k_folds}-Fold Cross Validation with Predefined Folds on {device}")
    print(f"Dataset size: {len(dataset)}")
    if standardize:
        print("Using standardization based on training data statistics")
    
    for fold, (train_ids, val_ids) in enumerate(fold_indices):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}")
        print(f"{'='*50}")
        
        # Apply standardization if requested
        if standardize:
            train_mean, train_std = compute_standardization_stats(dataset, train_ids)
            print(f"Training data statistics - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
            
            train_subset = create_standardized_subset(dataset, train_ids, train_mean, train_std)
            val_subset = create_standardized_subset(dataset, val_ids, train_mean, train_std)
        else:
            from torch.utils.data import Subset
            train_subset = Subset(dataset, train_ids)
            val_subset = Subset(dataset, val_ids)
        
        # Compute class weights for this fold if enabled
        if use_class_weights:
            # Extract training labels for this fold
            train_labels = [dataset[i][1].item() for i in train_ids]
            all_classes = np.arange(num_classes)
            present_classes = set(train_labels)
            missing_classes = set(all_classes) - present_classes

            if missing_classes:
                print(f"WARNING: Classes {missing_classes} are missing from training set in this fold. Disabling class weights for this fold.")
                criterion = nn.CrossEntropyLoss()
            else:
                class_weights_array = compute_class_weight(
                    'balanced',
                    classes=all_classes,
                    y=train_labels
                )
                class_weights = torch.tensor(class_weights_array, dtype=torch.float32).to(device)
                
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                print(f"Class weights computed: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Create data loaders
        if standardize:
            # For standardized data, we need custom DataLoader handling
            train_dataset = StandardizedDataset(train_subset)
            val_dataset = StandardizedDataset(val_subset)
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=12, pin_memory=True, persistent_workers=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=12, pin_memory=True, persistent_workers=True
            )
        else:
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=12, pin_memory=True, persistent_workers=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                num_workers=12, pin_memory=True, persistent_workers=True
            )
        
        # Initialize model and optimizer
        model = model_class(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2, min_lr=1e-6)
        
        # Train the fold
        fold_history = train_single_fold(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs, device, fold_num=fold+1, estop=estop, scheduler=scheduler
        )
        
        # Get final predictions if aggregating
        if aggregate_predictions:
            final_val_loss, final_val_acc, final_val_f1, final_preds, final_targets = validate_epoch(
                model, val_loader, criterion, device, return_predictions=True
            )
            all_final_predictions.append(final_preds)
            all_final_targets.append(final_targets)
        else:
            final_val_loss = fold_history['val_losses'][-1]
            final_val_acc = fold_history['val_accuracies'][-1]
            final_val_f1 = fold_history['val_f1s'][-1]
        
        # Calculate best values for this fold
        fold_best_acc = max(fold_history['val_accuracies'])
        fold_best_f1 = max(fold_history['val_f1s'])
        fold_best_loss = min(fold_history['val_losses'])
        
        # Store best results
        best_accs.append(fold_best_acc)
        best_f1s.append(fold_best_f1)
        best_losses.append(fold_best_loss)
        
        # Store fold results
        fold_results[f'fold_{fold+1}'] = {
            'history': fold_history,
            'final_val_acc': final_val_acc,
            'final_val_loss': final_val_loss,
            'final_val_f1': final_val_f1,
            'best_val_acc': fold_best_acc,
            'best_val_f1': fold_best_f1,
            'model_state': model.state_dict().copy(),
            'class_weights': class_weights.cpu() if use_class_weights and 'class_weights' in locals() else None
        }
        
        if standardize:
            fold_results[f'fold_{fold+1}']['train_mean'] = train_mean.item()
            fold_results[f'fold_{fold+1}']['train_std'] = train_std.item()
        
        final_val_accuracies.append(final_val_acc)
        final_val_losses.append(final_val_loss)
        final_val_f1s.append(final_val_f1)
    
    # Calculate aggregate statistics
    if aggregate_predictions:
        # Compute true aggregated cross-entropy
        all_predictions = torch.cat(all_final_predictions, dim=0)
        all_targets = torch.cat(all_final_targets, dim=0)
        
        criterion_agg = nn.CrossEntropyLoss()
        aggregated_loss = criterion_agg(all_predictions, all_targets).item()
        
        # Compute aggregated accuracy and F1
        aggregated_preds = all_predictions.argmax(dim=1)
        aggregated_accuracy = (aggregated_preds == all_targets).float().mean().item()
        aggregated_f1 = f1_score(all_targets.numpy(), aggregated_preds.numpy(), average='macro', zero_division=0)
        
        summary = {
            'aggregated_accuracy': aggregated_accuracy,
            'aggregated_loss': aggregated_loss,
            'aggregated_f1': aggregated_f1,
            'mean_val_accuracy': np.mean(final_val_accuracies),
            'std_val_accuracy': np.std(final_val_accuracies),
            'mean_val_loss': np.mean(final_val_losses),
            'std_val_loss': np.std(final_val_losses),
            'mean_val_f1': np.mean(final_val_f1s),
            'std_val_f1': np.std(final_val_f1s),
            'individual_accuracies': final_val_accuracies,
            'individual_losses': final_val_losses,
            'individual_f1s': final_val_f1s
        }
    else:
        # Use mean of fold losses (original approach)
        mean_val_acc = np.mean(final_val_accuracies)
        std_val_acc = np.std(final_val_accuracies)
        mean_val_loss = np.mean(final_val_losses)
        std_val_loss = np.std(final_val_losses)
        mean_val_f1 = np.mean(final_val_f1s)
        std_val_f1 = np.std(final_val_f1s)
        
        summary = {
            'mean_val_accuracy': mean_val_acc,
            'std_val_accuracy': std_val_acc,
            'mean_val_loss': mean_val_loss,
            'std_val_loss': std_val_loss,
            'mean_val_f1': mean_val_f1,
            'std_val_f1': std_val_f1,
            'individual_accuracies': final_val_accuracies,
            'individual_losses': final_val_losses,
            'individual_f1s': final_val_f1s
        }
    
    # Compile results
    results = {
        'fold_results': fold_results,
        'summary': summary,
        'config': {
            'k_folds': k_folds,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'device': str(device),
            'aggregate_predictions': aggregate_predictions,
            'use_class_weights': use_class_weights,
            'predefined_folds': True
        }
    }
    
    # Create best results dictionary
    best_results = {
        'accuracies': best_accs,
        'f1s': best_f1s,
        'losses': best_losses
    }
    
    return results, best_results

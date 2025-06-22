# Training Engine - Core Training Execution Logic
# Handles the actual training loops, data loading, and model management

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm

from utils.dataset_utils import (
    compute_standardization_stats, 
    create_standardized_subset,
    create_augmented_dataset_wrapper
)
from utils.evaluation_utils import get_confusion_matrix


class TrainingEngine:
    """
    Core training engine that handles model training execution.
    Provides methods for both single-fold and cross-validation training.
    """
    
    def __init__(self, model_class, num_classes, config):
        """
        Initialize training engine.
        
        Args:
            model_class: PyTorch model class to instantiate
            num_classes (int): Number of output classes
            config (dict): Training configuration parameters
        """
        self.model_class = model_class
        self.num_classes = num_classes
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Training engine initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model: {model_class.__name__}")
        print(f"  Classes: {num_classes}")
    
    def run_cross_validation(self, dataset, fold_indices):
        """
        Execute K-fold cross-validation training.
        
        Args:
            dataset: PyTorch TensorDataset
            fold_indices: List of (train_indices, val_indices) tuples
        
        Returns:
            tuple: (results, best_results)
        """
        print(f"\\nStarting {len(fold_indices)}-fold cross-validation...")
        
        fold_results = {}
        final_val_accuracies = []
        final_val_losses = []
        final_val_f1s = []
        
        best_accs = []
        best_f1s = []
        best_losses = []
        
        # For aggregated predictions
        all_final_predictions = []
        all_final_targets = []
        
        for fold, (train_indices, val_indices) in enumerate(fold_indices):
            print(f"\\n{'='*50}")
            print(f"FOLD {fold + 1}/{len(fold_indices)}")
            print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
            print(f"{'='*50}")
            
            # Train single fold
            fold_result = self._train_single_fold_with_indices(
                dataset, train_indices, val_indices, fold_num=fold+1
            )
            
            # Extract metrics
            history = fold_result['history']
            final_val_acc = fold_result['final_val_acc']
            final_val_loss = fold_result['final_val_loss']
            final_val_f1 = fold_result['final_val_f1']
            
            # Store results
            fold_results[f'fold_{fold+1}'] = fold_result
            final_val_accuracies.append(final_val_acc)
            final_val_losses.append(final_val_loss)
            final_val_f1s.append(final_val_f1)
            
            # Best results
            best_accs.append(max(history['val_accuracies']))
            best_f1s.append(max(history['val_f1s']))
            best_losses.append(min(history['val_losses']))
            
            # Aggregate predictions if enabled
            if self.config.get('aggregate_predictions', True):
                all_final_predictions.append(fold_result['val_predictions'])
                all_final_targets.append(fold_result['val_targets'])
        
        # Calculate summary statistics
        summary = self._calculate_cv_summary(
            final_val_accuracies, final_val_losses, final_val_f1s,
            all_final_predictions, all_final_targets
        )
        
        results = {
            'fold_results': fold_results,
            'summary': summary,
            'config': self.config.copy()
        }
        
        best_results = {
            'accuracies': best_accs,
            'f1s': best_f1s,
            'losses': best_losses
        }
        
        return results, best_results
    
    def run_single_fold_predefined(self, dataset, train_indices, val_indices):
        """
        Execute single fold training with predefined train/validation split.
        
        Args:
            dataset: PyTorch TensorDataset
            train_indices: Training set indices
            val_indices: Validation set indices
        
        Returns:
            dict: Training results
        """
        print(f"\\nStarting single fold training with predefined split...")
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
        
        return self._train_single_fold_with_indices(dataset, train_indices, val_indices)
    
    def run_single_fold_stratified(self, dataset):
        """
        Execute single fold training with stratified split.
        
        Args:
            dataset: PyTorch TensorDataset
        
        Returns:
            dict: Training results
        """
        print(f"\\nStarting single fold training with stratified split...")
        
        # Extract labels for stratification
        labels = [dataset[i][1].item() for i in range(len(dataset))]
        indices = list(range(len(dataset)))
        
        # Create stratified split
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=labels
        )
        
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
        
        return self._train_single_fold_with_indices(dataset, train_indices, val_indices)
    
    def _train_single_fold_with_indices(self, dataset, train_indices, val_indices, fold_num=None):
        """
        Internal method to train a single fold with given indices.
        
        Args:
            dataset: PyTorch TensorDataset
            train_indices: Training set indices
            val_indices: Validation set indices
            fold_num: Fold number (for logging)
        
        Returns:
            dict: Complete training results
        """
        start_time = time.time()
        
        # Create data subsets
        train_subset, val_subset = self._create_data_subsets(
            dataset, train_indices, val_indices
        )
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(train_subset, val_subset)
        
        # Initialize model and training components
        model, criterion, optimizer, scheduler = self._initialize_training_components(
            train_indices, dataset
        )
        
        # Train the model
        history = self._train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, fold_num
        )
        
        # Get final validation metrics
        final_val_loss, final_val_acc, final_val_f1, val_predictions, val_targets = self._validate_model(
            model, val_loader, criterion, return_predictions=True
        )
        
        # Generate confusion matrix
        val_confusion_matrix, _, _ = get_confusion_matrix(
            model, val_loader, self.device, self.num_classes
        )
        
        training_time = time.time() - start_time
        
        # Compile results
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
            'training_time': training_time,
            'config': self.config.copy()
        }
        
        # Log completion
        fold_str = f"Fold {fold_num} " if fold_num else ""
        print(f"\\n{fold_str}Training Complete!")
        if history['early_stopped']:
            print(f"Early stopped after {history['total_epochs']} epochs (best at epoch {history['best_epoch'] + 1})")
        print(f"Final - Val Acc: {final_val_acc:.4f}, Val Loss: {final_val_loss:.4f}, Val F1: {final_val_f1:.4f}")
        print(f"Best - Val Acc: {results['best_val_acc']:.4f}, Val F1: {results['best_val_f1']:.4f}")
        print(f"Training time: {training_time:.1f} seconds")
        
        return results
    
    def _create_data_subsets(self, dataset, train_indices, val_indices):
        """Create standardized data subsets if requested."""
        if self.config.get('standardize', False):
            print("Computing standardization statistics...")
            train_mean, train_std = compute_standardization_stats(
                dataset, train_indices, 
                sample_size=self.config.get('standardize_sample_size', 1000)
            )
            print(f"Standardization stats - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
            
            train_subset = create_standardized_subset(dataset, train_indices, train_mean, train_std)
            val_subset = create_standardized_subset(dataset, val_indices, train_mean, train_std)
            
            return train_subset, val_subset
        else:
            return Subset(dataset, train_indices), Subset(dataset, val_indices)
    
    def _create_data_loaders(self, train_subset, val_subset):
        """Create optimized data loaders."""
        # Determine optimal worker count
        num_workers = min(4, os.cpu_count() or 1) if self.config.get('standardize', False) else 0
        
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        return train_loader, val_loader
    
    def _initialize_training_components(self, train_indices, dataset):
        """Initialize model, criterion, optimizer, and scheduler."""
        # Initialize model
        model = self.model_class(num_classes=self.num_classes).to(self.device)
        
        # Initialize criterion with class weights if requested
        criterion = nn.CrossEntropyLoss()
        if self.config.get('use_class_weights', False):
            class_weights = self._compute_class_weights(train_indices, dataset)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Initialize optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.2, min_lr=1e-6
        )
        
        return model, criterion, optimizer, scheduler
    
    def _compute_class_weights(self, train_indices, dataset):
        """Compute balanced class weights for training."""
        print("Computing class weights...")
        train_labels = [dataset[i][1].item() for i in train_indices]
        unique_classes = np.unique(train_labels)
        
        class_weights_array = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=train_labels
        )
        
        class_weights = torch.ones(self.num_classes)
        for i, cls in enumerate(unique_classes):
            class_weights[cls] = class_weights_array[i]
        
        class_weights = class_weights.to(self.device)
        print(f"Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
        
        return class_weights
    
    def _train_model(self, model, train_loader, val_loader, criterion, optimizer, scheduler, fold_num):
        """Execute the main training loop."""
        history = {
            'train_losses': [], 'train_accuracies': [], 'train_f1s': [],
            'val_losses': [], 'val_accuracies': [], 'val_f1s': [],
            'learning_rates': [], 'early_stopped': False, 'best_epoch': 0, 'total_epochs': 0
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        estop = self.config.get('early_stopping', 35)
        
        fold_desc = f"Fold {fold_num}" if fold_num else "Training"
        
        for epoch in tqdm(range(self.config['num_epochs']), desc=f"{fold_desc}"):
            # Training phase
            train_loss, train_acc, train_f1 = self._train_epoch(
                model, train_loader, criterion, optimizer
            )
            
            # Validation phase
            val_loss, val_acc, val_f1 = self._validate_model(
                model, val_loader, criterion
            )
            
            # Record metrics
            history['train_losses'].append(train_loss)
            history['train_accuracies'].append(train_acc)
            history['train_f1s'].append(train_f1)
            history['val_losses'].append(val_loss)
            history['val_accuracies'].append(val_acc)
            history['val_f1s'].append(val_f1)
            
            # Learning rate scheduling
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
            
            # Progress update
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.config['num_epochs']} - "
                      f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f} - "
                      f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
        
        return history
    
    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """Train model for one epoch."""
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
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
    
    def _validate_model(self, model, val_loader, criterion, return_predictions=False):
        """Validate model and return metrics."""
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_targets = [], []
        all_predictions, all_target_tensors = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
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
            return (val_loss / val_total, val_correct / val_total, f1, 
                   torch.cat(all_predictions), torch.cat(all_target_tensors))
        else:
            return val_loss / val_total, val_correct / val_total, f1
    
    def _calculate_cv_summary(self, accuracies, losses, f1s, predictions=None, targets=None):
        """Calculate cross-validation summary statistics."""
        summary = {
            'mean_val_accuracy': np.mean(accuracies),
            'std_val_accuracy': np.std(accuracies),
            'mean_val_loss': np.mean(losses),
            'std_val_loss': np.std(losses),
            'mean_val_f1': np.mean(f1s),
            'std_val_f1': np.std(f1s),
            'individual_accuracies': accuracies,
            'individual_losses': losses,
            'individual_f1s': f1s
        }
        
        # Add aggregated metrics if predictions available
        if predictions and targets and self.config.get('aggregate_predictions', True):
            all_predictions = torch.cat(predictions, dim=0)
            all_targets = torch.cat(targets, dim=0)
            
            criterion_agg = nn.CrossEntropyLoss()
            aggregated_loss = criterion_agg(all_predictions, all_targets).item()
            
            aggregated_preds = all_predictions.argmax(dim=1)
            aggregated_accuracy = (aggregated_preds == all_targets).float().mean().item()
            aggregated_f1 = f1_score(
                all_targets.numpy(), aggregated_preds.numpy(), 
                average='macro', zero_division=0
            )
            
            summary.update({
                'aggregated_accuracy': aggregated_accuracy,
                'aggregated_loss': aggregated_loss,
                'aggregated_f1': aggregated_f1
            })
        
        return summary

# Training Engine - Core Training Execution Logic
# Handles the actual training loops, data loading, and model management

import os
import time
import threading
import queue
import gc
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from utils.evaluation_utils import get_confusion_matrix
from utils.specaugment import get_augmentation_params
from utils.dataloader_factory import OptimalDataLoaderFactory, DataLoaderConfigLogger
from utils.dataset_utils import (
    compute_standardization_stats, 
    create_standardized_subset,
    create_augmented_dataset_wrapper)

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
        print(f"\\nStarting single fold training...")
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
        
        # Use the existing single fold training logic
        return self._train_single_fold_with_indices(
            dataset, train_indices, val_indices, fold_num=1
        )
    
    def run_single_fold_stratified(self, dataset):
        """
        Execute single fold training with stratified split.
        
        Args:
            dataset: PyTorch TensorDataset
        
        Returns:
            dict: Training results
        """
        
        # Extract labels for stratification
        labels = dataset.tensors[1].numpy()
        indices = np.arange(len(dataset))
        
        # Stratified split
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=self.config.get('test_size', 0.2),
            stratify=labels,
            random_state=self.config.get('random_state', 42)
        )
        
        print(f"\\nStarting single fold training (stratified split)...")
        print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
        
        # Use the existing single fold training logic
        return self._train_single_fold_with_indices(
            dataset, train_indices, val_indices, fold_num=1
        )

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
        if history['early_stopped']:
            print(f"Early stopped after {history['total_epochs']} epochs (best at epoch {history['best_epoch'] + 1})")
        print(f"Best - Val Acc: {results['best_val_acc']:.4f}, Val F1: {results['best_val_f1']:.4f}")
        
        return results
    
    def _create_data_subsets(self, dataset, train_indices, val_indices):
        """Create standardized and/or augmented data subsets if requested."""
        # First apply standardization if requested
        if self.config.get('standardize', False):
            train_mean, train_std = compute_standardization_stats(
                dataset, train_indices, 
                sample_size=self.config.get('standardize_sample_size', 1000)
            )
            
            train_subset = create_standardized_subset(dataset, train_indices, train_mean, train_std)
            val_subset = create_standardized_subset(dataset, val_indices, train_mean, train_std)
        else:
            from torch.utils.data import Subset
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
        
        # Then apply augmentation to training set only if requested
        use_spec_augment = self.config.get('spec_augment', False)
        use_gaussian_noise = self.config.get('gaussian_noise', False)
        
        if use_spec_augment or use_gaussian_noise:            
            # Get augmentation parameters
            augment_params = get_augmentation_params(
                len(train_indices), 
                self.num_classes,
                aggressive=self.config.get('aggressive_augmentation', False)
            )
            
            # Apply augmentation only to training set
            train_subset = create_augmented_dataset_wrapper(
                train_subset, 
                use_spec_augment=use_spec_augment,
                use_gaussian_noise=use_gaussian_noise,
                augment_params=augment_params,
                training=True
            )
            
            # Validation set remains unaugmented
            val_subset = create_augmented_dataset_wrapper(
                val_subset,
                use_spec_augment=False,
                use_gaussian_noise=False,
                training=False
            )        
        return train_subset, val_subset
    
    def _create_data_loaders(self, train_subset, val_subset):
        """Create optimized data loaders using the factory."""
        # Determine dataset characteristics
        has_augmentation = self.config.get('spec_augment', False) or self.config.get('gaussian_noise', False)
        has_standardization = self.config.get('standardize', False)
        
        # # Log configuration for debugging
        # if self.config.get('debug_dataloaders', False):
        #     print(f"\\nDataLoader Configuration:")
        #     print(f"  Has augmentation: {has_augmentation}")
        #     print(f"  Has standardization: {has_standardization}")
        #     print(f"  Train dataset size: {len(train_subset)}")
        #     print(f"  Val dataset size: {len(val_subset)}")
        
        train_loader = OptimalDataLoaderFactory.create_training_loader(
            train_subset,
            batch_size=self.config['batch_size'],
            has_augmentation=has_augmentation,
            has_standardization=has_standardization
        )
        
        val_loader = OptimalDataLoaderFactory.create_validation_loader(
            val_subset,
            batch_size=self.config['batch_size'],
            has_standardization=has_standardization
        )
        
        # Log configurations if debugging enabled
        if self.config.get('debug_dataloaders', False):
            DataLoaderConfigLogger.log_config("Training", train_loader.__dict__, len(train_subset))
            DataLoaderConfigLogger.log_config("Validation", val_loader.__dict__, len(val_subset))
        
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
        
        # Initialize optimizer - use config to decide between Adam and SGD
        use_adam = self.config.get('use_adam', True)  # Default to Adam if not specified
        learning_rate = self.config.get('learning_rate', self.config.get('initial_lr', 0.001))
        l2_reg = self.config.get('l2_regularization', 0)
        
        if use_adam:
            optimizer = optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=l2_reg
            )
        else:
            momentum = self.config.get('momentum', 0.9)  # Allow configurable momentum
            optimizer = optim.SGD(
                model.parameters(), 
                lr=learning_rate,
                momentum=momentum,
                weight_decay=l2_reg
            )
        
        # Initialize scheduler based on config
        scheduler = self._create_scheduler(optimizer)
        
        return model, criterion, optimizer, scheduler
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler based on configuration."""
        lr_schedule = self.config.get('lr_schedule', None)
        
        if lr_schedule is None:
            return None
        
        if isinstance(lr_schedule, dict):
            schedule_type = lr_schedule.get('type')
            
            if schedule_type == 'plateau':
                return ReduceLROnPlateau(
                    optimizer, 
                    mode='min',
                    patience=lr_schedule.get('patience', 10),
                    factor=lr_schedule.get('factor', 0.2),
                    min_lr=lr_schedule.get('min_lr', 1e-6)
                )
            elif schedule_type == 'exponential':
                return ExponentialLR(
                    optimizer,
                    gamma=lr_schedule.get('gamma', 0.95)
                )
            elif schedule_type == 'cosine':
                return CosineAnnealingLR(
                    optimizer,
                    T_max=lr_schedule.get('T_max', 50),
                    eta_min=lr_schedule.get('eta_min', 1e-6)
                )
            else:
                raise ValueError(f"Unknown lr_schedule type: {schedule_type}")
        else:
            print(f"Warning: lr_schedule should be a dict, got {type(lr_schedule)}. Using ReduceLROnPlateau as default.")
            return ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.2, min_lr=1e-6
            )
    
    def _compute_class_weights(self, train_indices, dataset):
        """Compute balanced class weights for training."""
        # print("Computing class weights...")
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
        # print(f"Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
        
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
        estop = self.config.get('estop_thresh', self.config.get('early_stopping', 35))
        best_model_state = None
        
        fold_desc = f"Fold {fold_num}" if fold_num else "Training"
        config_id = self.config.get('config_id', 'Unknown')
        
        # Initialize timing for progress estimates
        start_time = time.time()
        
        # Create progress bar with detailed information
        progress_bar = tqdm(range(self.config['num_epochs']), desc=f"{fold_desc}")
        
        for epoch in progress_bar:
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
            
            # Calculate time estimates
            elapsed_time = time.time() - start_time
            epochs_completed = epoch + 1
            avg_time_per_epoch = elapsed_time / epochs_completed
            remaining_epochs = self.config['num_epochs'] - epochs_completed
            estimated_total_time = elapsed_time + (remaining_epochs * avg_time_per_epoch)
            
            # Format time strings
            elapsed_str = f"{int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}"
            total_str = f"{int(estimated_total_time//60):02d}:{int(estimated_total_time%60):02d}"
            
            # Update progress bar with detailed information
            progress_info = (
                f"Config: {config_id} | "
                f"TrLoss: {train_loss:.4f} | TrAcc: {train_acc:.4f} | "
                f"ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.4f}"
            )
            progress_bar.set_description(progress_info)
            
            # Learning rate scheduling - handle different scheduler types
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    # For other schedulers (ExponentialLR, CosineAnnealingLR), step without metric
                    scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Early stopping check with model state saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history['best_epoch'] = epoch
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= estop:
                    # print(f"Early stopping at epoch {epoch + 1}")
                    # print(f"Restoring best model from epoch {history['best_epoch'] + 1}")
                    # Restore best model state
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    history['early_stopped'] = True
                    break
            
            history['total_epochs'] = epoch + 1
        
        # Close progress bar
        progress_bar.close()
        
        # Always restore best model at the end (if not early stopped)
        if best_model_state is not None and not history['early_stopped']:
            model.load_state_dict(best_model_state)
            print(f"Training completed. Restored best model from epoch {history['best_epoch'] + 1}")
        
        return history
    
    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """Train model for one epoch."""
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        
        # Mixed precision training components
        use_amp = self.config.get('mixed_precision', True)  # Default enabled for RTX 5080
        gradient_clipping = self.config.get('gradient_clipping', 1.0)  # Default clip value
        
        if use_amp and hasattr(torch.cuda, 'amp'):
            scaler = torch.amp.GradScaler()
        else:
            scaler = None
            use_amp = False
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            optimizer.zero_grad()
            
            if use_amp and scaler is not None:
                # Mixed precision forward pass
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping (on scaled gradients)
                if gradient_clipping > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
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

    def run_cross_validation_parallel(self, dataset, fold_indices, max_parallel_folds=2):
        """
        Execute K-fold cross-validation training with TRUE GPU parallel processing.
        
        Uses threading with CUDA streams for efficient GPU utilization on high-end hardware.
        Automatically falls back to sequential training if CUDA is not available.
        Optimized for RTX 5080 with 16GB VRAM.
        
        Args:
            dataset: PyTorch TensorDataset
            fold_indices: List of (train_indices, val_indices) tuples
            max_parallel_folds (int): Maximum number of concurrent fold training threads
        
        Returns:
            tuple: (results, best_results)
        """
        
        print(f"\nStarting {len(fold_indices)}-fold cross-validation with TRUE GPU PARALLEL processing...")
        print(f"Maximum concurrent folds: {max_parallel_folds}")
        
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available! Falling back to sequential CPU training.")
            return self.run_cross_validation(dataset, fold_indices)
        
        # GPU memory information
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_memory_free = gpu_memory_total - gpu_memory_allocated
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {gpu_memory_free:.1f}GB free / {gpu_memory_total:.1f}GB total")
        print(f"Estimated memory per fold: ~{gpu_memory_free / max_parallel_folds:.1f}GB")
        
        # Create CUDA streams for parallel execution
        streams = [torch.cuda.Stream() for _ in range(max_parallel_folds)]
        
        # Results containers
        results_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def train_fold_gpu_parallel(fold_idx, train_indices, val_indices, stream):
            """
            Train a single fold on GPU with dedicated CUDA stream.
            """
                # Set the stream for this thread
            with torch.cuda.stream(stream):
                # Create isolated training engine for this fold
                fold_engine = TrainingEngine(
                    model_class=self.model_class,
                    num_classes=self.num_classes,
                    config=self.config.copy()
                )
                fold_engine.device = self.device  # Keep GPU device
                
                print(f"\n🚀 Starting Fold {fold_idx + 1}/{len(fold_indices)} on GPU (Stream {stream.stream_id})")
                
                # Train the fold with stream isolation
                fold_result = fold_engine._train_single_fold_with_indices(
                    dataset, train_indices, val_indices, fold_num=fold_idx+1
                )
                
                # Clean up model from results to save memory
                if 'model' in fold_result:
                    # Save model state but remove the model object
                    fold_result['model_state_dict'] = fold_result['model'].state_dict().copy()
                    del fold_result['model']
                
                # Synchronize stream before cleanup
                stream.synchronize()
                
                # Clean up GPU memory for this stream
                torch.cuda.empty_cache()
                
                print(f"✅ Fold {fold_idx + 1} completed successfully on GPU")
                results_queue.put((fold_idx, fold_result))
        
        # Execute folds in parallel using threading
        print(f"\n🔥 Launching {max_parallel_folds} concurrent GPU training threads...")
        
        threads = []
        fold_batches = []
        
        # Group folds into batches for parallel execution
        for i in range(0, len(fold_indices), max_parallel_folds):
            batch = fold_indices[i:i + max_parallel_folds]
            fold_batches.append(batch)
        
        # Process each batch of folds in parallel
        all_fold_results = {}
        
        for batch_idx, batch in enumerate(fold_batches):
            print(f"\n📦 Processing batch {batch_idx + 1}/{len(fold_batches)} ({len(batch)} folds)")
            
            # Launch threads for this batch
            batch_threads = []
            for thread_idx, (train_indices, val_indices) in enumerate(batch):
                fold_idx = batch_idx * max_parallel_folds + thread_idx
                stream = streams[thread_idx % len(streams)]
                
                thread = threading.Thread(
                    target=train_fold_gpu_parallel,
                    args=(fold_idx, train_indices, val_indices, stream),
                    name=f"FoldThread-{fold_idx+1}"
                )
                thread.start()
                batch_threads.append(thread)
            
            # Wait for all threads in this batch to complete
            for thread in batch_threads:
                thread.join()
            
            # Collect results from this batch
            while not results_queue.empty():
                fold_idx, result = results_queue.get()
                all_fold_results[fold_idx] = result
            
            # Check for errors
            batch_errors = []
            while not error_queue.empty():
                fold_idx, error = error_queue.get()
                batch_errors.append((fold_idx, error))
            
            if batch_errors:
                for fold_idx, error in batch_errors:
                    print(f"❌ Fold {fold_idx + 1} failed: {error}")
            
            print(f"✅ Batch {batch_idx + 1} completed")
            
            # GPU memory cleanup between batches
            torch.cuda.empty_cache()
            gc.collect()
        
        # Process and aggregate all results
        fold_results = {}
        final_val_accuracies = []
        final_val_losses = []
        final_val_f1s = []
        best_accs = []
        best_f1s = []
        best_losses = []
        all_final_predictions = []
        all_final_targets = []
        
        # Sort results by fold index
        sorted_fold_results = sorted(all_fold_results.items())
        
        for fold_idx, fold_result in sorted_fold_results:
            if 'error' not in fold_result:
                # Extract metrics
                history = fold_result['history']
                final_val_acc = fold_result['final_val_acc']
                final_val_loss = fold_result['final_val_loss']
                final_val_f1 = fold_result['final_val_f1']
                
                # Store results
                fold_results[f'fold_{fold_idx+1}'] = fold_result
                final_val_accuracies.append(final_val_acc)
                final_val_losses.append(final_val_loss)
                final_val_f1s.append(final_val_f1)
                
                # Best results
                best_accs.append(max(history['val_accuracies']))
                best_f1s.append(max(history['val_f1s']))
                best_losses.append(min(history['val_losses']))
                
                # Aggregate predictions if enabled
                if self.config.get('aggregate_predictions', True):
                    if 'val_predictions' in fold_result and 'val_targets' in fold_result:
                        all_final_predictions.append(fold_result['val_predictions'])
                        all_final_targets.append(fold_result['val_targets'])
        
        # Calculate summary statistics
        if final_val_accuracies:
            summary = self._calculate_cv_summary(
                final_val_accuracies, final_val_losses, final_val_f1s,
                all_final_predictions, all_final_targets
            )
        else:
            print("❌ ERROR: All folds failed during parallel execution!")
            summary = {'error': 'All folds failed'}
        
        # Final GPU memory cleanup
        for stream in streams:
            stream.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        results = {
            'fold_results': fold_results,
            'summary': summary,
            'config': self.config.copy(),
            'parallel_execution': True,
            'gpu_parallel': True,  # Flag to indicate true GPU parallel execution
            'max_parallel_folds': max_parallel_folds,
            'cuda_streams_used': len(streams)
        }
        
        best_results = {
            'accuracies': best_accs,
            'f1s': best_f1s,
            'losses': best_losses
        }
        
        print(f"\n🎉 TRUE GPU PARALLEL cross-validation completed!")
        print(f"✅ Successfully completed {len(final_val_accuracies)} out of {len(fold_indices)} folds")
        print(f"🚀 Used {len(streams)} CUDA streams for parallel execution")
        
        return results, best_results
    
    def _monitor_gpu_memory(self, prefix=""):
        """Monitor and log GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - allocated
            
            print(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free / {total:.2f}GB total")
            
            # Warning if memory usage is high
            if allocated > total * 0.8:
                print(f"⚠️  WARNING: High GPU memory usage ({allocated/total*100:.1f}%)!")
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': free,
                'total_gb': total,
                'usage_percent': (allocated / total) * 100
            }
        return None

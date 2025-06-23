# Evaluation and Visualization Utilities
# Handles confusion matrices, metrics visualization, and result analysis

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Evaluation Functions
def get_confusion_matrix(model, data_loader, device, num_classes):
    """
    Generate confusion matrix from model predictions.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation data
        device: Computing device (CPU/GPU)
        num_classes: Number of classes
    
    Returns:
        tuple: (confusion_matrix, predictions, targets)
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_predictions_tensor = []
    all_targets_tensor = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_predictions_tensor.append(outputs.cpu())
            all_targets_tensor.append(y_batch.cpu())
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    
    # Concatenate tensors for return
    predictions_tensor = torch.cat(all_predictions_tensor, dim=0)
    targets_tensor = torch.cat(all_targets_tensor, dim=0)
    
    return cm, predictions_tensor, targets_tensor

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", figsize=(12, 10)):
    """
    Plot confusion matrix showing only percentages to avoid overcrowding.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names (optional)
        title: Plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create heatmap
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.1f',
                cmap='Blues',
                square=True,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks + 0.5, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks + 0.5, class_names, rotation=0)
    
    plt.tight_layout()
    plt.show()
    plt.close()  # Free figure memory

# Result Visualization Functions
def plot_best_results(best_results, metric_key, title, ylabel, ax=None):
    """
    Plot best results across folds.
    
    Args:
        best_results: Dictionary containing best results for each metric
        metric_key: Key for the metric to plot ('accuracies', 'f1s', 'losses')
        title: Plot title
        ylabel: Y-axis label
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    values = best_results[metric_key]
    folds = range(1, len(values) + 1)
    
    # Plot individual fold results
    ax.plot(folds, values, 'bo-', linewidth=2, markersize=8, alpha=0.7, label='Individual Folds')
    
    # Plot mean line
    mean_val = np.mean(values)
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.4f}')
    
    # Add standard deviation band
    std_val = np.std(values)
    ax.fill_between(folds, mean_val - std_val, mean_val + std_val, 
                    alpha=0.2, color='red', label=f'±1 STD: {std_val:.4f}')
    
    ax.set_xlabel('Fold')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(folds)
    
    # Add value annotations
    for fold, value in zip(folds, values):
        ax.annotate(f'{value:.3f}', (fold, value), 
                   textcoords="offset points", xytext=(0,10), ha='center')

def plot_mean_curve(results, metric_key, title, ylabel, ax=None):
    """
    Plot mean training curves across folds.
    
    Args:
        results: Results dictionary from cross-validation
        metric_key: Key for the metric to plot
        title: Plot title
        ylabel: Y-axis label
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect all fold curves
    all_curves = []
    fold_results = results['fold_results']
    
    for fold_name, fold_data in fold_results.items():
        history = fold_data['history']
        if metric_key in history and history[metric_key]:
            all_curves.append(history[metric_key])
    
    if not all_curves:
        ax.text(0.5, 0.5, f"No data available for {metric_key}", 
               transform=ax.transAxes, ha='center', va='center')
        return
    
    # Find minimum length to align all curves
    min_length = min(len(curve) for curve in all_curves)
    aligned_curves = [curve[:min_length] for curve in all_curves]
    
    # Convert to numpy array for easier manipulation
    curves_array = np.array(aligned_curves)
    
    # Calculate statistics
    mean_curve = np.mean(curves_array, axis=0)
    std_curve = np.std(curves_array, axis=0)
    epochs = range(1, len(mean_curve) + 1)
    
    # Plot individual curves (lighter)
    for curve in aligned_curves:
        ax.plot(epochs, curve, alpha=0.3, color='gray', linewidth=1)
    
    # Plot mean curve
    ax.plot(epochs, mean_curve, color='blue', linewidth=3, label='Mean')
    
    # Add standard deviation band
    ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                    alpha=0.2, color='blue', label='±1 STD')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_kfold_results(results, best_results):
    """
    Create comprehensive visualization of K-fold cross-validation results.
    
    Args:
        results: Results dictionary from cross-validation
        best_results: Best results dictionary
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Best results plots
    plot_best_results(best_results, 'accuracies', 'Best Validation Accuracy per Fold', 
                     'Accuracy', axes[0, 0])
    plot_best_results(best_results, 'f1s', 'Best Validation F1 Score per Fold', 
                     'F1 Score', axes[0, 1])
    plot_best_results(best_results, 'losses', 'Best Validation Loss per Fold', 
                     'Loss', axes[0, 2])
      # Mean curves plots
    plot_mean_curve(results, 'val_accuracies', 'Mean Validation Accuracy Curves', 
                   'Accuracy', axes[1, 0])
    plot_mean_curve(results, 'val_f1s', 'Mean Validation F1 Score Curves', 
                   'F1 Score', axes[1, 1])
    plot_mean_curve(results, 'val_losses', 'Mean Validation Loss Curves', 
                   'Loss', axes[1, 2])
    
    plt.tight_layout()
    plt.show()
    plt.close()  # Free figure memory

def plot_single_fold_curve(results, metric_key, title, ylabel):
    """
    Plot training curves for a single fold.
    
    Args:
        results: Results dictionary from single fold training
        metric_key: Key for the metric to plot
        title: Plot title
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))
    
    history = results['history']
    epochs = range(1, len(history[metric_key]) + 1)
    
    plt.plot(epochs, history[metric_key], linewidth=2, label=metric_key.replace('_', ' ').title())
    
    # Mark best epoch if available
    if 'best_epoch' in history:
        best_epoch = history['best_epoch']
        best_value = history[metric_key][best_epoch]
        plt.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best Epoch: {best_epoch + 1}')
        plt.scatter([best_epoch + 1], [best_value], color='red', s=100, zorder=5)
    
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()  # Free figure memory

# Results Printing Functions
def print_single_fold_results(results):
    """Print detailed results from single fold training."""
    print("\n" + "="*60)
    print("SINGLE FOLD TRAINING RESULTS")
    print("="*60)
    
    print(f"Final Validation Accuracy: {results['final_val_acc']:.4f}")
    print(f"Final Validation Loss: {results['final_val_loss']:.4f}")
    print(f"Final Validation F1: {results['final_val_f1']:.4f}")
    print(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
    print(f"Best Validation F1: {results['best_val_f1']:.4f}")
    
    if 'config' in results:
        config = results['config']
        print(f"\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

def print_confusion_matrix_stats(results):
    """Print statistics from confusion matrix."""
    if 'confusion_matrix' not in results:
        print("No confusion matrix available in results.")
        return
    
    cm = results['confusion_matrix']
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX STATISTICS")
    print("="*60)
    
    # Overall accuracy
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    overall_accuracy = total_correct / total_samples
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    # Per-class statistics
    print(f"\nPer-class Statistics:")
    print(f"{'Class':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 50)
    
    for class_idx in range(cm.shape[0]):
        # Precision: TP / (TP + FP)
        tp = cm[class_idx, class_idx]
        fp = np.sum(cm[:, class_idx]) - tp
        fn = np.sum(cm[class_idx, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(cm[class_idx, :])
        
        print(f"{class_idx:<6} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<8}")

def print_kfold_best_results(results):
    """Print best results from K-fold cross-validation."""
    print("\n" + "="*60)
    print("K-FOLD CROSS-VALIDATION BEST RESULTS")
    print("="*60)
    
    summary = results['summary']
    
    if 'aggregated_accuracy' in summary:
        print("AGGREGATED METRICS (across all folds):")
        print(f"  Aggregated Accuracy: {summary['aggregated_accuracy']:.4f}")
        print(f"  Aggregated Loss: {summary['aggregated_loss']:.4f}")
        print(f"  Aggregated F1 Score: {summary['aggregated_f1']:.4f}")
        print()
    
    print("MEAN METRICS (average of fold results):")
    print(f"  Mean Validation Accuracy: {summary['mean_val_accuracy']:.4f} ± {summary['std_val_accuracy']:.4f}")
    print(f"  Mean Validation Loss: {summary['mean_val_loss']:.4f} ± {summary['std_val_loss']:.4f}")
    print(f"  Mean Validation F1: {summary['mean_val_f1']:.4f} ± {summary['std_val_f1']:.4f}")
    
    print(f"\nINDIVIDUAL FOLD RESULTS:")
    individual_accs = summary['individual_accuracies']
    individual_f1s = summary['individual_f1s']
    individual_losses = summary['individual_losses']
    
    for i, (acc, f1, loss) in enumerate(zip(individual_accs, individual_f1s, individual_losses)):
        print(f"  Fold {i+1}: Acc={acc:.4f}, F1={f1:.4f}, Loss={loss:.4f}")
    
    if 'config' in results:
        config = results['config']
        print(f"\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

# Model Utils
def save_model(model, model_name, model_save_path):
    """Save model state dictionary to file."""
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to: {model_save_path}")

def test_saved_model(save_path):
    """Test loading a saved model and print basic info."""
    state = torch.load(save_path, map_location='cpu')
    print(type(state))
    print(list(state.keys())[:5])  # show first 5 parameter names
    print(state[list(state.keys())[0]].shape)  # show shape of first tensor

def load_model(model_class, model_name, num_classes=29):
    """Load a saved model from file."""
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=num_classes).to(device)
    model_path = os.path.join('..', 'models', f"{model_name}.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def reset_model(model_class, lr=0.001, num_classes=29):
    """Reset and initialize a new model with optimizer and criterion."""
    import torch.optim as optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion, device

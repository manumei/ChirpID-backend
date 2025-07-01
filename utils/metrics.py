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

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", figsize=(12, 10), show_counts=False):
    """
    Plot confusion matrix showing percentages and optionally counts.
    
    Args:
        cm: Confusion matrix array (numpy array)
        class_names: List of class names (optional)
        title: Plot title
        figsize: Figure size tuple
        show_counts: If True, show both counts and percentages
    """
    # Convert to numpy array if it's a tensor
    if hasattr(cm, 'numpy'):
        cm = cm.numpy()
    elif hasattr(cm, 'cpu'):
        cm = cm.cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Prepare annotations
    if show_counts:
        # Show both counts and percentages
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = cm_normalized[i, j]
                if count > 0:
                    row.append(f'{count}\n({percentage:.1f}%)')
                else:
                    row.append('0\n(0.0%)')
            annotations.append(row)
        
        # Create heatmap with custom annotations
        sns.heatmap(cm_normalized, 
                    annot=np.array(annotations), 
                    fmt='',
                    cmap='Blues',
                    square=True,
                    cbar_kws={'label': 'Percentage (%)'})
    else:
        # Show only percentages
        sns.heatmap(cm_normalized, 
                    annot=True, 
                    fmt='.1f',
                    cmap='Blues',
                    square=True,
                    cbar_kws={'label': 'Percentage (%)'})
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks + 0.5, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks + 0.5, class_names, rotation=0)
    
    plt.tight_layout()
    plt.show()
    plt.close()  # Free figure memory

def plot_metric_vs_epochs(train_values, val_values, metric_name, ax=None):
    """
    Helper function to plot training and validation metrics over epochs.
    
    Args:
        train_values: List of training metric values
        val_values: List of validation metric values
        metric_name: Name of the metric for labeling
        ax: Matplotlib axis to plot on (if None, creates new figure)
    """
    epochs = range(1, len(train_values) + 1)
    
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    
    ax.plot(epochs, train_values, 'b-', label=f'Training {metric_name}', linewidth=2)
    ax.plot(epochs, val_values, 'r-', label=f'Validation {metric_name}', linewidth=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_full_metrics(config_id, history: dict, cm: np.ndarray):
    """ Plots all the relevant metrics from the training history.
    Plots a 2x2 grid of subplots: Losses, Accuracies, F1 Scores and Confusion Matrix

    Args:
        config_id: Configuration identifier for the plot title
        history (dict): Dictionary containing training history with the keys:
            - 'train_losses': List of training losses
            - 'val_losses': List of validation losses
            - 'train_accuracies': List of training accuracies
            - 'val_accuracies': List of validation accuracies
            - 'train_f1s': List of training F1 scores
            - 'val_f1s': List of validation F1 scores
        cm (np.ndarray): Confusion matrix to be plotted
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Metrics of {config_id}', fontsize=16, fontweight='bold')
    
    # [0, 0] Plot training and validation losses over epochs
    plot_metric_vs_epochs(history['train_losses'], history['val_losses'], 'Loss', axes[0, 0])
    
    # [0, 1] Plot training and validation accuracies over epochs
    plot_metric_vs_epochs(history['train_accuracies'], history['val_accuracies'], 'Accuracy', axes[0, 1])
    
    # [1, 0] Plot training and validation F1 scores over epochs
    plot_metric_vs_epochs(history['train_f1s'], history['val_f1s'], 'F1 Score', axes[1, 0])
    
    # [1, 1] Plot confusion matrix
    # Convert to numpy if needed
    if hasattr(cm, 'numpy'):
        cm = cm.numpy()
    elif hasattr(cm, 'cpu'):
        cm = cm.cpu().numpy()
    
    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create heatmap on the subplot
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.1f',
                cmap='Blues',
                square=True,
                cbar_kws={'label': 'Percentage (%)'},
                ax=axes[1, 1])
    
    axes[1, 1].set_title('Confusion Matrix', fontsize=12)
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()
    plt.close()  # Free figure memory

def plot_metrics(config_id, results):
    history = results.get('history', {})
    conf_matrix = results.get('confusion_matrix', None)
    
    if not history or not conf_matrix:
        raise ValueError(f"Results must contain 'history' and 'confusion_matrix' keys. Currently got: {results}")
    
    plot_full_metrics(config_id, history, conf_matrix)

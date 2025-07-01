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

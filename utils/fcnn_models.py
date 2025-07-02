import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

def relu(x):
    return np.maximum(0, x)

def dreludx(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred), axis=1)

def tv_split(dset,tproporcion,semilla):
    np.random.seed(semilla)
    idxs=np.arange(dset.shape[0])
    np.random.shuffle(idxs)
    split=int(np.ceil(tproporcion*dset.shape[0]))
    t = dset.iloc[idxs[:split]].copy()
    v = dset.iloc[idxs[split:]].copy()
    return t,v

def plot_confusion_matrix(y_true, y_pred, num_classes):
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=False, fmt="d", cmap="viridis")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()



class BirdFCNN(nn.Module):
    ''' Fully Connected Neural Network using PyTorch '''
    def __init__(self, num_classes, input_dim=70112, hidden_layers=[512, 128, 32], dropout_p=0.5):
        super(BirdFCNN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            return probabilities

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
            return predictions

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=100, 
                    lr=0.001, batch_size=32, optimizer_type='adam', 
                    l2_lambda=0.0, early_stopping=True, patience=10, 
                    eval_interval=10, lr_schedule=None, class_weights=None, device='cpu'):
        
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)

        if valX is not None and valY is not None:
            valX = torch.tensor(valX, dtype=torch.float32, device=device)
            valY = torch.tensor(valY, dtype=torch.long, device=device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_lambda) if optimizer_type == 'adam' else torch.optim.SGD(self.parameters(), lr=lr, weight_decay=l2_lambda)

        best_val_loss = float('inf')
        wait = 0

        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []
        train_f1_history, val_f1_history = [], []

        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            
            for i in range(0, trainX.size(0), batch_size):
                idx = indices[i:i+batch_size]
                X_batch = trainX[idx]
                y_batch = trainY[idx]

                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if lr_schedule and lr_schedule.get('type') == 'exponential':
                decay = lr_schedule.get('decay', 0.96)
                for g in optimizer.param_groups:
                    g['lr'] = lr * (decay ** epoch)

            if epoch % eval_interval == 0:
                self.eval()
                with torch.no_grad():
                    # Training metrics
                    outputs = self(trainX)
                    train_loss = criterion(outputs, trainY).item()
                    preds = torch.argmax(outputs, dim=1)
                    train_acc = (preds == trainY).float().mean().item()
                    train_f1 = f1_score(trainY.cpu().numpy(), preds.cpu().numpy(), average='weighted')

                    train_loss_history.append(train_loss)
                    train_acc_history.append(train_acc)
                    train_f1_history.append(train_f1)

                    if valX is not None and valY is not None:
                        val_outputs = self(valX)
                        val_loss = criterion(val_outputs, valY).item()
                        val_preds = torch.argmax(val_outputs, dim=1)
                        val_acc = (val_preds == valY).float().mean().item()
                        val_f1 = f1_score(valY.cpu().numpy(), val_preds.cpu().numpy(), average='weighted')

                        val_loss_history.append(val_loss)
                        val_acc_history.append(val_acc)
                        val_f1_history.append(val_f1)
                        
                    if epoch % 30 == 0:
                        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - F1: {train_f1:.4f}", end="")
                        print(f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")
                    else:
                        print()

                if early_stopping and valX is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = self.state_dict()
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            self.load_state_dict(best_model_state)
                            break

        # Generate final confusion matrix on validation set
        self.eval()
        with torch.no_grad():
            if valX is not None and valY is not None:
                val_outputs = self(valX)
                val_preds = torch.argmax(val_outputs, dim=1)
                cm = confusion_matrix(valY.cpu().numpy(), val_preds.cpu().numpy())
            else:
                # Use training set if no validation set
                outputs = self(trainX)
                preds = torch.argmax(outputs, dim=1)
                cm = confusion_matrix(trainY.cpu().numpy(), preds.cpu().numpy())

        # Prepare history dictionary for plot_full_metrics
        history = {
            'train_losses': train_loss_history,
            'val_losses': val_loss_history,
            'train_accuracies': train_acc_history,
            'val_accuracies': val_acc_history,
            'train_f1s': train_f1_history,
            'val_f1s': val_f1_history
        }

        return {
            'history': history,
            'confusion_matrix': cm,
            'best_val_f1': max(val_f1_history) if val_f1_history else max(train_f1_history),
            'best_val_acc': max(val_acc_history) if val_acc_history else max(train_acc_history)
        }

    def save_model(self, filepath):
        """Save the entire model to a file"""
        torch.save(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """Load a model from a file"""
        model = torch.load(filepath, weights_only=False)
        print(f"Model loaded from {filepath}")
        return model

class BirdFCNN_v2(BirdFCNN):
    """Deep narrow network - more layers, smaller hidden units"""
    def __init__(self, num_classes, input_dim=70112, dropout_p=0.3):
        super(BirdFCNN_v2, self).__init__(
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_layers=[256, 128, 64, 32, 16],
            dropout_p=dropout_p
        )
    
    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=100, 
                    lr=0.0005, batch_size=16, optimizer_type='adam', 
                    l2_lambda=1e-5, early_stopping=True, patience=10, 
                    eval_interval=10, lr_schedule={'type': 'exponential', 'decay': 0.95}, 
                    class_weights=None, device='cpu'):
        """Deep network with lower LR and exponential decay"""
        return super().train_model(
            trainX=trainX, trainY=trainY, valX=valX, valY=valY,
            epochs=epochs, lr=lr, batch_size=batch_size, 
            optimizer_type=optimizer_type, l2_lambda=l2_lambda,
            early_stopping=early_stopping, patience=patience,
            eval_interval=eval_interval, lr_schedule=lr_schedule,
            class_weights=class_weights, device=device
        )

class BirdFCNN_v3(BirdFCNN):
    """Wide shallow network - fewer layers, larger hidden units"""
    def __init__(self, num_classes, input_dim=70112, dropout_p=0.5):
        super(BirdFCNN_v3, self).__init__(
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_layers=[1024, 512],
            dropout_p=dropout_p
        )
    
    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=100, 
                    lr=0.002, batch_size=64, optimizer_type='adam', 
                    l2_lambda=1e-3, early_stopping=True, patience=10, 
                    eval_interval=10, lr_schedule=None,
                    class_weights=None, device='cpu'):
        """Wide network with higher LR and no schedule"""
        return super().train_model(
            trainX=trainX, trainY=trainY, valX=valX, valY=valY,
            epochs=epochs, lr=lr, batch_size=batch_size, 
            optimizer_type=optimizer_type, l2_lambda=l2_lambda,
            early_stopping=early_stopping, patience=patience,
            eval_interval=eval_interval, lr_schedule=lr_schedule,
            class_weights=class_weights, device=device
        )

class BirdFCNN_v4(BirdFCNN):
    """Low dropout network - same as original but with minimal dropout"""
    def __init__(self, num_classes, input_dim=70112, dropout_p=0.1):
        super(BirdFCNN_v4, self).__init__(
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_layers=[512, 128, 32],
            dropout_p=dropout_p
        )
    
    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=100, 
                    lr=0.001, batch_size=128, optimizer_type='sgd', 
                    l2_lambda=1e-4, early_stopping=True, patience=10, 
                    eval_interval=10, lr_schedule={'type': 'linear'},
                    class_weights=None, device='cpu'):
        """Low dropout with SGD and linear LR decay"""
        return super().train_model(
            trainX=trainX, trainY=trainY, valX=valX, valY=valY,
            epochs=epochs, lr=lr, batch_size=batch_size, 
            optimizer_type=optimizer_type, l2_lambda=l2_lambda,
            early_stopping=early_stopping, patience=patience,
            eval_interval=eval_interval, lr_schedule=lr_schedule,
            class_weights=class_weights, device=device
        )

class BirdFCNN_v5(BirdFCNN):
    """Progressive shrinking network"""
    def __init__(self, num_classes, input_dim=70112, dropout_p=0.4):
        super(BirdFCNN_v5, self).__init__(
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_layers=[1024, 256, 64, 16],
            dropout_p=dropout_p
        )
    
    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=100, 
                    lr=0.003, batch_size=32, optimizer_type='adam', 
                    l2_lambda=5e-4, early_stopping=True, patience=10, 
                    eval_interval=10, lr_schedule={'type': 'exponential', 'decay': 0.98},
                    class_weights=None, device='cpu'):
        """Progressive network with aggressive LR and moderate decay"""
        return super().train_model(
            trainX=trainX, trainY=trainY, valX=valX, valY=valY,
            epochs=epochs, lr=lr, batch_size=batch_size, 
            optimizer_type=optimizer_type, l2_lambda=l2_lambda,
            early_stopping=early_stopping, patience=patience,
            eval_interval=eval_interval, lr_schedule=lr_schedule,
            class_weights=class_weights, device=device
        )

class BirdFCNN_v6(BirdFCNN):
    """Very deep network"""
    def __init__(self, num_classes, input_dim=70112, dropout_p=0.2):
        super(BirdFCNN_v6, self).__init__(
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_layers=[512, 256, 128, 64, 32, 16, 8],
            dropout_p=dropout_p
        )
    
    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=100, 
                    lr=0.0001, batch_size=8, optimizer_type='adam', 
                    l2_lambda=1e-6, early_stopping=True, patience=10, 
                    eval_interval=10, lr_schedule={'type': 'exponential', 'decay': 0.99},
                    class_weights=None, device='cpu'):
        """Very deep network with very low LR and small batches"""
        return super().train_model(
            trainX=trainX, trainY=trainY, valX=valX, valY=valY,
            epochs=epochs, lr=lr, batch_size=batch_size, 
            optimizer_type=optimizer_type, l2_lambda=l2_lambda,
            early_stopping=early_stopping, patience=patience,
            eval_interval=eval_interval, lr_schedule=lr_schedule,
            class_weights=class_weights, device=device
        )

class BirdFCNN_v7(BirdFCNN):
    """Minimal network - very simple architecture"""
    def __init__(self, num_classes, input_dim=70112, dropout_p=0.6):
        super(BirdFCNN_v7, self).__init__(
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_layers=[128],
            dropout_p=dropout_p
        )
    
    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=100, 
                    lr=0.01, batch_size=256, optimizer_type='sgd', 
                    l2_lambda=1e-2, early_stopping=True, patience=10, 
                    eval_interval=10, lr_schedule=None,
                    class_weights=None, device='cpu'):
        """Minimal network with high LR, large batches, and SGD"""
        return super().train_model(
            trainX=trainX, trainY=trainY, valX=valX, valY=valY,
            epochs=epochs, lr=lr, batch_size=batch_size, 
            optimizer_type=optimizer_type, l2_lambda=l2_lambda,
            early_stopping=early_stopping, patience=patience,
            eval_interval=eval_interval, lr_schedule=lr_schedule,
            class_weights=class_weights, device=device
        )

class BirdFCNN_v0(BirdFCNN):
    """Original baseline network - standard configuration with SGD optimizer"""
    def __init__(self, num_classes, input_dim=70112, dropout_p=0.5):
        super(BirdFCNN_v0, self).__init__(
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_layers=[512, 256, 128],
            dropout_p=dropout_p
        )
    
    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=100, 
                    lr=0.01, batch_size=64, optimizer_type='sgd', 
                    l2_lambda=1e-3, early_stopping=True, patience=10, 
                    eval_interval=10, lr_schedule=None, class_weights=None, device='cpu'):
        """Original baseline with SGD and standard parameters"""
        return super().train_model(
            trainX=trainX, trainY=trainY, valX=valX, valY=valY,
            epochs=epochs, lr=lr, batch_size=batch_size, 
            optimizer_type=optimizer_type, l2_lambda=l2_lambda,
            early_stopping=early_stopping, patience=patience,
            eval_interval=eval_interval, lr_schedule=lr_schedule,
            class_weights=class_weights, device=device
        )

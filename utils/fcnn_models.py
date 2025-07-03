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
                    eval_interval=10, lr_schedule=None, class_weights=None, device='cuda'):
        
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
                    eval_interval=10, lr_schedule=None, class_weights=None, device='cuda'):
        """Original baseline with SGD and standard parameters"""
        return super().train_model(
            trainX=trainX, trainY=trainY, valX=valX, valY=valY,
            epochs=epochs, lr=lr, batch_size=batch_size, 
            optimizer_type=optimizer_type, l2_lambda=l2_lambda,
            early_stopping=early_stopping, patience=patience,
            eval_interval=eval_interval, lr_schedule=lr_schedule,
            class_weights=class_weights, device=device
        )


class BirdFCNN_v1(nn.Module):
    """Residual-inspired FCNN with skip connections"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v1, self).__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.skip = nn.Linear(1024, 128)
        self.classifier = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.input_norm(x)
        x1 = F.gelu(self.bn1(self.fc1(x)))
        x1 = self.dropout(x1)
        x2 = F.gelu(self.bn2(self.fc2(x1)))
        x2 = self.dropout(x2)
        x3 = F.gelu(self.bn3(self.fc3(x2)))
        x3 = self.dropout(x3)
        x4 = F.gelu(self.bn4(self.fc4(x3)))
        skip = self.skip(x1)
        x4 = x4 + skip
        return self.classifier(x4)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=150, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 64):
                idx = indices[i:i+64]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

class BirdFCNN_v2(nn.Module):
    """Wide network with spectral normalization"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v2, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 2048)),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.utils.spectral_norm(nn.Linear(2048, 1024)),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(1024, 512)),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=200, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        class_weights = torch.tensor([1.2, 0.8, 1.0, 1.1, 0.9], device=device)  # Example weights
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005, betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 32):
                idx = indices[i:i+32]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

class BirdFCNN_v3(nn.Module):
    """Deep narrow network with progressive dropout"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v3, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(input_dim, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16)
        ])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(dim) for dim in [512, 256, 128, 64, 32, 16]])
        self.dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        for i, (fc, bn) in enumerate(zip(self.fc_layers, self.bn_layers)):
            x = fc(x)
            x = bn(x)
            x = F.leaky_relu(x, 0.01)
            x = F.dropout(x, self.dropouts[i], training=self.training)
        return self.classifier(x)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=120, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.002, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 16):
                idx = indices[i:i+16]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

class BirdFCNN_v4(nn.Module):
    """Attention-based FCNN"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v4, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(attn_out.squeeze(1) + x.squeeze(1))
        return self.classifier(x)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=100, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0008, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 24):
                idx = indices[i:i+24]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

class BirdFCNN_v5(nn.Module):
    """Ensemble-style multi-path network"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v5, self).__init__()
        # Path 1: Deep narrow
        self.path1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        # Path 2: Wide shallow
        self.path2 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(0.4),
            nn.Linear(1024, 64)
        )
        self.combiner = nn.Linear(128, num_classes)

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        combined = torch.cat([p1, p2], dim=1)
        return self.combiner(combined)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=180, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0015, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, epochs=180, steps_per_epoch=len(trainX)//48)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 48):
                idx = indices[i:i+48]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

class BirdFCNN_v6(nn.Module):
    """Regularization-heavy network"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v6, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 800),
            nn.GroupNorm(8, 800),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(800, 400),
            nn.GroupNorm(8, 400),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(400, 200),
            nn.GroupNorm(8, 200),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(200, 100),
            nn.GroupNorm(4, 100),
            nn.ELU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=140, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        class_weights = torch.tensor([0.8, 1.3, 1.0, 1.1, 0.9], device=device)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, cycle_momentum=False)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 128):
                idx = indices[i:i+128]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

class BirdFCNN_v7(nn.Module):
    """Minimal but optimized network"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v7, self).__init__()
        self.norm_input = nn.BatchNorm1d(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.Swish(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Swish(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.norm_input(x)
        return self.net(x)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=160, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.003, weight_decay=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.7)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            total_loss = 0
            for i in range(0, trainX.size(0), 256):
                idx = indices[i:i+256]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step(total_loss)

class BirdFCNN_v8(nn.Module):
    """Stochastic depth network"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v8, self).__init__()
        self.blocks = nn.ModuleList([
            self._make_block(input_dim, 1024),
            self._make_block(1024, 512),
            self._make_block(512, 256),
            self._make_block(256, 128),
        ])
        self.classifier = nn.Linear(128, num_classes)
        self.survival_probs = [0.9, 0.8, 0.7, 0.6]

    def _make_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if self.training and torch.rand(1) > self.survival_probs[i]:
                continue
            x = block(x)
        return self.classifier(x)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=200, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 32):
                idx = indices[i:i+32]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

class BirdFCNN_v9(nn.Module):
    """Mixup-enabled network"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v9, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1536),
            nn.LayerNorm(1536),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def mixup_data(self, x, y, alpha=0.4):
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=150, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0012, weight_decay=3e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 64):
                idx = indices[i:i+64]
                x_batch, y_batch = trainX[idx], trainY[idx]
                mixed_x, y_a, y_b, lam = self.mixup_data(x_batch, y_batch)
                outputs = self(mixed_x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

class BirdFCNN_v10(nn.Module):
    """Temperature-scaled network with calibration"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v10, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 896),
            nn.BatchNorm1d(896),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(896, 448),
            nn.BatchNorm1d(448),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(448, 224),
            nn.BatchNorm1d(224),
            nn.SiLU(),
            nn.Linear(224, num_classes)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x, calibrate=False):
        logits = self.backbone(x)
        if calibrate:
            return logits / self.temperature
        return logits

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=120, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        class_weights = torch.tensor([1.1, 0.9, 1.2, 0.8, 1.0], device=device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0018, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 40):
                idx = indices[i:i+40]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                optimizer.step()
            scheduler.step()

class BirdFCNN_v11(nn.Module):
    """Variational dropout network"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v11, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 1200),
            nn.Linear(1200, 600),
            nn.Linear(600, 300),
            nn.Linear(300, 150),
            nn.Linear(150, num_classes)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(1200),
            nn.LayerNorm(600),
            nn.LayerNorm(300),
            nn.LayerNorm(150)
        ])
        # Different dropout rates for each layer
        self.dropout_rates = [0.3, 0.4, 0.5, 0.6]

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, self.dropout_rates[i], training=self.training)
        return self.layers[-1](x)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=180, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0008, eps=1e-7)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=45, T_mult=1.5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 20):
                idx = indices[i:i+20]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

class BirdFCNN_v12(nn.Module):
    """Knowledge distillation ready network"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v12, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Mish()
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=140, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0025, weight_decay=1e-4, betas=(0.95, 0.999))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.008, epochs=140, steps_per_epoch=len(trainX)//80)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 80):
                idx = indices[i:i+80]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

class BirdFCNN_v13(nn.Module):
    """Self-supervised pre-training ready network"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v13, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1600),
            nn.GroupNorm(16, 1600),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(1600, 800),
            nn.GroupNorm(16, 800),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(800, 400),
            nn.GroupNorm(8, 400),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(400, 200)
        )
        self.head = nn.Linear(200, num_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.head(encoded)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=250, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        class_weights = torch.tensor([0.9, 1.2, 1.0, 1.3, 0.8], device=device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0005, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 12):
                idx = indices[i:i+12]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 2.0)
                optimizer.step()
            scheduler.step()

class BirdFCNN_v14(nn.Module):
    """Extreme regularization network for small datasets"""
    def __init__(self, num_classes, input_dim=70112):
        super(BirdFCNN_v14, self).__init__()
        self.input_dropout = nn.Dropout(0.1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(96, num_classes)
        )

    def forward(self, x):
        x = self.input_dropout(x)
        return self.net(x)

    def train_model(self, trainX, trainY, valX=None, valY=None, epochs=300, device='cuda'):
        self.to(device)
        trainX = torch.tensor(trainX, dtype=torch.float32, device=device)
        trainY = torch.tensor(trainY, dtype=torch.long, device=device)
        if valX is not None: valX = torch.tensor(valX, dtype=torch.float32, device=device)
        if valY is not None: valY = torch.tensor(valY, dtype=torch.long, device=device)
        
        class_weights = torch.tensor([1.0, 1.1, 0.9, 1.2, 1.0], device=device)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.95, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.3)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
        
        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(trainX.size(0))
            for i in range(0, trainX.size(0), 8):
                idx = indices[i:i+8]
                outputs = self(trainX[idx])
                loss = criterion(outputs, trainY[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                optimizer.step()
            scheduler.step()


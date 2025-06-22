import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


class Nn:
    def __init__(self, trainX, trainY, m, seed):
        """
        X: matriz de imágenes (n samples x p pixeles)
        Y: matriz one-hot de etiquetas (n samples, c clases)
        m: lista con cantidad de nodos por capa oculta (no incluye input ni output)

        Esta versión redefine `m` para incluir entrada, ocultas y salida.
        """
        self.X = trainX
        self.y = trainY
        # Redefinimos `m` para incluir todas las capas (entrada, ocultas, salida)
        m = [trainX.shape[1]] + m + [trainY.shape[1]]                                   # m[0] es el tamaño de la entrada
        self.L = len(m) - 1       # cantidad de capas con pesos                         # m[L] es el tamaño de la salida

        np.random.seed(seed)
        self.W = [None]*(self.L+1)             # lista de matrices W
        self.b = [None]*(self.L+1)             # lista de matrices b
        for l in range(1,self.L+1):             # iterar L veces
            fan_in = m[l-1]
            fan_out = m[l]
            limit = np.sqrt(6 / (fan_in + fan_out))  # Glorot uniform
            self.W[l]=(np.random.uniform(-limit, limit, (fan_out, fan_in)))
            self.b[l]=(np.zeros(fan_out))

        # Initialize history tracking
        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []

    def ff(self,X=None):
        Z = [None]*(self.L+1)              # lista de matrices z
        A = [None]*(self.L+1)              # lista de matrices a
        if X is None:
            Z[0] = self.X     
        else:
            Z[0] = X            # para el caso de usar la red para predecir validation o test        
        for l in range(1,self.L+1):
            A[l] = (Z[l-1] @ self.W[l].T) + self.b[l]
            if l==self.L:
                Z[l] = softmax(A[l])
            else:
                Z[l] = relu(A[l])
        return A,Z
    
    def bp(self, A, Z,y_true=None):
        if y_true is None:
            y_true = self.y
        delta = [None] * (self.L + 1)
        delta[self.L] = Z[self.L] - y_true
        dLossdW = [None] * (self.L + 1)
        dLossdW[self.L] = delta[self.L].T @ Z[self.L - 1]
        for l in range(self.L - 1, 0, -1):
            delta[l] = (delta[l + 1] @ self.W[l + 1]) * dreludx(A[l])
            dLossdW[l] = delta[l].T @ Z[l - 1]
        return delta, dLossdW
    
    def trainUltimate(self, 
          epochs, 
          lr, 
          batch_size=None, 
          optimizer='sgd', 
          l2_lambda=0.0, 
          valX=None, 
          valy=None, 
          early_stopping=False, 
          patience=None, 
          eval_interval=10, 
          lr_schedule=None):

        n_samples = self.X.shape[0]
        best_val_loss = float('inf')
        best_weights = None
        best_biases = None
        wait = 0
        
        # Initialize history lists
        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []

        # For ADAM
        if optimizer == 'adam':
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            m_W = [np.zeros_like(w) if w is not None else None for w in self.W]
            v_W = [np.zeros_like(w) if w is not None else None for w in self.W]
            m_b = [np.zeros_like(b) if b is not None else None for b in self.b]
            v_b = [np.zeros_like(b) if b is not None else None for b in self.b]

        for epoch in range(epochs):
            # Learning rate schedule
            if lr_schedule and lr_schedule.get('type') == 'exponential':
                decay = lr_schedule.get('decay', 0.96)
                lr_epoch = lr * (decay ** epoch)
            elif lr_schedule and lr_schedule.get('type') == 'linear':
                lr_epoch = lr * (1 - epoch / epochs)
            else:
                lr_epoch = lr

            indices = np.random.permutation(n_samples)
            batches = [(indices[i:i + batch_size] if batch_size else indices)
                    for i in range(0, n_samples, batch_size or n_samples)]

            for batch_idx in batches:
                X_batch = self.X[batch_idx]
                y_batch = self.y[batch_idx]

                A, Z = self.ff(X_batch)
                delta, dLossdW = self.bp(A, Z, y_batch)

                for l in range(1, self.L + 1):
                    grad_W = dLossdW[l] + l2_lambda * self.W[l]
                    grad_b = np.mean(delta[l], axis=0)

                    if optimizer == 'adam':
                        m_W[l] = beta1 * m_W[l] + (1 - beta1) * grad_W
                        v_W[l] = beta2 * v_W[l] + (1 - beta2) * (grad_W ** 2)
                        m_hat_W = m_W[l] / (1 - beta1 ** (epoch + 1))
                        v_hat_W = v_W[l] / (1 - beta2 ** (epoch + 1))
                        self.W[l] -= lr_epoch * m_hat_W / (np.sqrt(v_hat_W) + epsilon)

                        m_b[l] = beta1 * m_b[l] + (1 - beta1) * grad_b
                        v_b[l] = beta2 * v_b[l] + (1 - beta2) * (grad_b ** 2)
                        m_hat_b = m_b[l] / (1 - beta1 ** (epoch + 1))
                        v_hat_b = v_b[l] / (1 - beta2 ** (epoch + 1))
                        self.b[l] -= lr_epoch * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
                    else:  # SGD
                        self.W[l] -= lr_epoch * grad_W
                        self.b[l] -= lr_epoch * grad_b

            # Full forward pass for tracking loss and accuracy
            _, Z_full = self.ff()
            train_loss = np.mean(cross_entropy(self.y, Z_full[self.L]))
            self.loss_history.append(train_loss)
            
            # Calculate training accuracy
            train_predictions = np.argmax(Z_full[self.L], axis=1)
            train_true = np.argmax(self.y, axis=1)
            train_accuracy = np.mean(train_predictions == train_true)
            self.accuracy_history.append(train_accuracy)

            if valX is not None and valy is not None:         #trackear loss del validation set 
                _, Z_val = self.ff(valX)
                val_loss = np.mean(cross_entropy(valy, Z_val[self.L]))
                self.val_loss_history.append(val_loss)
                
                # Calculate validation accuracy
                val_predictions = np.argmax(Z_val[self.L], axis=1)
                val_true = np.argmax(valy, axis=1)
                val_accuracy = np.mean(val_predictions == val_true)
                self.val_accuracy_history.append(val_accuracy)

                # Early stopping check
                if patience is not None and epoch % eval_interval == 0:    #patience is not None o tambien puede ser early_stopping=False
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = [w.copy() if w is not None else None for w in self.W]
                        best_biases = [b.copy() if b is not None else None for b in self.b]
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f"Early stopping at epoch {epoch + 1}")
                            self.W = best_weights
                            self.b = best_biases
                            break
            else:
                if epoch % eval_interval == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f}")

        # Plot training history
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, label="Train Loss")
        if valX is not None and valy is not None:
            plt.plot(range(1, len(self.val_loss_history) + 1), self.val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot accuracies
        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(self.accuracy_history) + 1), self.accuracy_history, label="Train Accuracy")
        if valX is not None and valy is not None:
            plt.plot(range(1, len(self.val_accuracy_history) + 1), self.val_accuracy_history, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate over time if using schedule
        plt.subplot(1, 3, 3)
        if lr_schedule and lr_schedule.get('type') == 'exponential':
            decay = lr_schedule.get('decay', 0.96)
            lr_values = [lr * (decay ** epoch) for epoch in range(len(self.loss_history))]
            plt.plot(range(1, len(lr_values) + 1), lr_values, label="Learning Rate")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.legend()
            plt.grid(True)
        else:
            # Plot final accuracies as bars if no LR schedule
            if valX is not None and valy is not None:
                final_train_acc = self.accuracy_history[-1] if self.accuracy_history else 0
                final_val_acc = self.val_accuracy_history[-1] if self.val_accuracy_history else 0
                plt.bar(['Training', 'Validation'], [final_train_acc, final_val_acc], 
                       alpha=0.7, color=['blue', 'red'])
                plt.ylabel('Final Accuracy')
                plt.title('Final Training vs Validation Accuracy')
                plt.ylim(0, 1)
                for i, acc in enumerate([final_train_acc, final_val_acc]):
                    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
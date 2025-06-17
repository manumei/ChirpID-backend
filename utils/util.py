# Directory Aux
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def clean_dir(dest_dir):
    ''' Deletes the raw audio files in the dest_dir.'''
    print(f"Resetting {dest_dir} directory...")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

def count_files_in_dir(dir_path):
    ''' Counts the number of files in a directory.'''
    if not os.path.exists(dir_path):
        return 0
    return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

# Audio Processing Aux
import librosa as lbrs
import noisereduce as nr
from PIL import Image
import numpy as np

def lbrs_loading(audio_path, sr, mono=True):
    y, srate = lbrs.load(audio_path, sr=sr, mono=mono)
    if srate != sr:
        raise ValueError(f"Sample rate mismatch: expected {sr}, got {srate}, at audio file {audio_path}")
    return y, srate

def get_rmsThreshold(y, frame_len, hop_len, thresh_factor=0.5):
    rms = lbrs.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    threshold = thresh_factor * np.mean(rms)
    return threshold

def reduce_noise_seg(segment, srate, filename, class_id):
    try:
        segment = nr.reduce_noise(y=segment, sr=srate, stationary=False)
    except RuntimeWarning as e:
        print(f"RuntimeWarning while reducing noise for segment in {filename} from {class_id}: {e}")
    except Exception as e:
        print(f"Error while reducing noise for segment in {filename} from {class_id}: {e}")
    return segment

def get_spec_norm(segment, sr, mels, hoplen, nfft):
    spec = lbrs.feature.melspectrogram(y=segment, sr=sr, n_mels=mels, hop_length=hoplen, n_fft=nfft)
    spec_db = lbrs.power_to_db(spec, ref=np.max)
    norm_spec = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
    return norm_spec

def get_spec_image(segment, sr, mels, hoplen, nfft, filename, start, spectrogram_dir):
    norm_spec = get_spec_norm(segment, sr, mels, hoplen, nfft)
    img = (norm_spec * 255).astype(np.uint8)
    spec_filename = f"{os.path.splitext(filename)[0]}_{start}.png"
    spec_path = os.path.join(spectrogram_dir, spec_filename)
    return img, spec_path, spec_filename

def save_test_audios(segment, sr, test_audios_dir, filename, start, saved_audios):
    if test_audios_dir is not None and saved_audios < 10:
        import soundfile as sf
        os.makedirs(test_audios_dir, exist_ok=True)
        test_audio_filename = f"{os.path.splitext(filename)[0]}_{start}_test.wav"
        test_audio_path = os.path.join(test_audios_dir, test_audio_filename)
        sf.write(test_audio_path, segment, sr)

# Data Processing Aux
def get_spect_matrix(image_path):
    img = Image.open(image_path).convert('L')
    pixels = np.array(img)
    return pixels

def get_spec_matrix_direct(segment, sr, mels, hoplen, nfft):
    """ Get spectrogram matrix directly from segment and params """
    norm_spec = get_spec_norm(segment, sr, mels, hoplen, nfft)
    matrix = (norm_spec * 255).astype(np.uint8)
    return matrix

def audio_process(audio_path, reduce_noise: bool, sr=32000, segment_sec=5.0,
                frame_len=2048, hop_len=512, mels=224, nfft=2048, thresh=0.75):
    ''' 
    Takes the path to an audio file (any format) and processes it to finally return 
    the list of grayscale spectrogram pixel matrices for each of its high-RMS segments.

    Step 1: Load the audio file with librosa. (using lbrs_loading)
    Step 2: Split into high-RMS segments of 5 seconds. (using get_rmsThreshold)
    Step 3: Reduce noise for each segment if reduce_noise is True. (using reduce_noise_seg)
    Step 4: Generate a Spectrogram grayscale matrix for each segment. (using get_spec_mnatrix_direct)
    '''

    matrices = []
    print(f"Processing audio file: {audio_path}")
    samples_per_segment = int(sr * segment_sec)

    # Step 1
    y, srate = lbrs_loading(audio_path, sr)

    # Step 2
    threshold = get_rmsThreshold(y, frame_len, hop_len, thresh_factor=thresh)

    for start in range(0, len(y) - samples_per_segment + 1, samples_per_segment):
        segment = y[start:start + samples_per_segment]
        seg_rms = np.mean(lbrs.feature.rms(y=segment)[0])
        
        if seg_rms < threshold:
            # print(f"Segment at {start} has {seg_rms} RMS, below threshold {threshold}. Skipping...")
            continue

        # Step 3
        if reduce_noise:
            segment = reduce_noise_seg(segment, srate, os.path.basename(audio_path), class_id=None)

        # Step 4
        filename = os.path.basename(audio_path)
        matrix = get_spec_matrix_direct(segment, srate, mels, hop_len, nfft)
        matrices.append(matrix)
    
    print(f"Processed {len(matrices)} segments from {audio_path}")
    return matrices

# Training and Validation Functions
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
    
    if return_predictions:
        all_predictions = []
        all_target_tensors = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
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
                    num_epochs, device, fold_num=None):
    """Train model on a single fold and return training history including F1 scores."""
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []
    
    desc = f"Fold {fold_num}" if fold_num is not None else "Training"
    pbar = tqdm(range(num_epochs), desc=desc, unit="epoch")
    
    for epoch in pbar:
        # Training
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)
        
        # Validation
        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)
        
        pbar.set_description(f"{desc} - Epoch {epoch+1}/{num_epochs}")
        pbar.set_postfix(
            train_acc=f"{train_acc:.3f}", 
            train_loss=f"{train_loss:.4f}",
            train_f1=f"{train_f1:.3f}",
            val_acc=f"{val_acc:.3f}",
            val_loss=f"{val_loss:.4f}",
            val_f1=f"{val_f1:.3f}"
        )
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s
    }

def k_fold_cross_validation(dataset, model_class, num_classes, k_folds=5, 
                            num_epochs=300, batch_size=32, lr=0.001, 
                            random_state=42, aggregate_predictions=True):
    """
    Perform K-Fold Cross Validation training with F1 score reporting.
    
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
                                If False, use mean of individual fold losses
    
    Returns:
        Dictionary containing results for each fold and aggregated metrics including F1 scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize KFold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    # Store results for each fold
    fold_results = {}
    final_val_accuracies = []
    final_val_losses = []
    final_val_f1s = []
    
    # For aggregated predictions
    if aggregate_predictions:
        all_final_predictions = []
        all_final_targets = []
    
    print(f"Starting {k_folds}-Fold Cross Validation on {device}")
    print(f"Dataset size: {len(dataset)}")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}")
        print(f"{'='*50}")
        
        # Create data subsets
        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Initialize model, criterion, and optimizer
        model = model_class(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train the fold
        fold_history = train_single_fold(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs, device, fold_num=fold+1
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
        
        # Store fold results
        fold_results[f'fold_{fold+1}'] = {
            'history': fold_history,
            'final_val_acc': final_val_acc,
            'final_val_loss': final_val_loss,
            'final_val_f1': final_val_f1,
            'best_val_acc': max(fold_history['val_accuracies']),
            'best_val_f1': max(fold_history['val_f1s']),
            'model_state': model.state_dict().copy()  # Save best model if needed
        }
        
        final_val_accuracies.append(final_val_acc)
        final_val_losses.append(final_val_loss)
        final_val_f1s.append(final_val_f1)
        
        print(f"Fold {fold+1} Final - Val Acc: {final_val_acc:.4f}, "
            f"Val Loss: {final_val_loss:.4f}, Val F1: {final_val_f1:.4f}")
    
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
            'aggregate_predictions': aggregate_predictions
        }
    }
    
    return results

def plot_mean_curve(results, metric_key, title, ylabel):
    all_train = []
    all_val = []

    for fold_data in results['fold_results'].values():
        history = fold_data['history']
        all_train.append(history[f"train_{metric_key}"])
        all_val.append(history[f"val_{metric_key}"])

    # Convert to arrays for averaging
    all_train = np.array(all_train)
    all_val = np.array(all_val)

    mean_train = all_train.mean(axis=0)
    mean_val = all_val.mean(axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(mean_train, label="Mean Train", linestyle='--')
    plt.plot(mean_val, label="Mean Val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Model Utils
def save_model(model, model_name):
    model_dir = os.path.join('..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to: {model_save_path}")

def load_model(model_class, model_name, num_classes=28):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=num_classes).to(device)
    model_path = os.path.join('..', 'models', f"{model_name}.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def reset_model(model_class, lr=0.001, num_classes=28):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion, device


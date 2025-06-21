# Directory Aux
import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils.util import *

# Custom dataset class for standardized data (needed for multiprocessing)
class StandardizedDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

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

# Data Processing Utils
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

def save_audio_segments_to_disk(segments, segments_output_dir):
    """
    Save extracted audio segments to disk as .wav files.
    
    Args:
        segments (list): List of segment dictionaries from extract_audio_segments
        segments_output_dir (str): Directory to save the audio segment files
        
    Returns:
        pd.DataFrame: DataFrame with segment metadata for later use
    """
    import soundfile as sf
    
    os.makedirs(segments_output_dir, exist_ok=True)
    
    segment_records = []
    
    for i, segment_info in enumerate(segments):
        # Create filename for the segment
        base_filename = os.path.splitext(segment_info['original_filename'])[0]
        segment_filename = f"{base_filename}_{segment_info['segment_index']}.wav"
        segment_path = os.path.join(segments_output_dir, segment_filename)
        
        # Save audio segment to disk
        sf.write(segment_path, segment_info['audio_data'], segment_info['sr'])
        og_filename = segment_info['original_filename']
        og_audio_name = os.path.splitext(og_filename)[0] if og_filename else 'unknown'
        
        # Create record for CSV
        segment_records.append({
            'filename': segment_filename,
            'class_id': segment_info['class_id'],
            'author': segment_info['author'],
            'original_audio': og_audio_name,
            'segment_index': segment_info['segment_index'],
            'species_segments': segment_info['class_total_segments']
        })
    
    return pd.DataFrame(segment_records)

def load_audio_segments_from_disk(segments_csv_path, segments_dir, sr=32000):
    """
    Load audio segments from disk using metadata CSV.
    
    Args:
        segments_csv_path (str): Path to CSV file with segment metadata
        segments_dir (str): Directory containing the audio segment files
        sr (int): Target sampling rate
        
    Returns:
        list: List of segment dictionaries ready for spectrogram creation
    """
    import soundfile as sf
    
    # Read metadata
    segments_df = pd.read_csv(segments_csv_path)
    
    segments = []
    
    for _, row in segments_df.iterrows():
        segment_path = os.path.join(segments_dir, row['filename'])
        
        try:
            # Load audio data
            audio_data, file_sr = sf.read(segment_path)
            
            if file_sr != sr:
                print(f"Warning: Sample rate mismatch for {row['filename']}: expected {sr}, got {file_sr}")
                # Could resample here if needed              # Create segment dictionary
            segment_info = {
                'audio_data': audio_data,
                'class_id': row['class_id'],
                'author': row['author'],
                'original_filename': row['original_audio'] + '.wav',  # Reconstruct original filename
                'segment_index': row['segment_index'],
                'sr': file_sr,
                'class_total_segments': row['species_segments']
            }
            
            segments.append(segment_info)
            
        except Exception as e:
            print(f"Error loading segment {row['filename']}: {e}")
            continue
    
    print(f"Loaded {len(segments)} audio segments from disk")
    return segments

def load_audio_files(segments_df, segments_dir, sr, segment_sec, threshold_factor):
    """Load and prepare audio files with metadata."""
    audio_files = []
    samples_per_segment = int(sr * segment_sec)
    
    for _, row in segments_df.iterrows():
        filename = row['filename']
        class_id = row['class_id']
        author = row['author']
        audio_path = os.path.join(segments_dir, filename)
        
        try:
            y, srate = lbrs_loading(audio_path, sr=sr, mono=True)
            threshold = get_rmsThreshold(y, frame_len=2048, hop_len=512, thresh_factor=threshold_factor)
            max_segments = len(y) // samples_per_segment
            
            if max_segments > 0:
                audio_files.append({
                    'audio_data': y,
                    'class_id': class_id,
                    'author': author,
                    'filename': filename,
                    'max_segments': max_segments,
                    'threshold': threshold,
                    'sr': srate,
                    'samples_per_segment': samples_per_segment
                })
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    return audio_files

def calculate_class_totals(audio_files):
    """Calculate total possible segments per class."""
    class_total_segments = {}
    for audio_info in audio_files:
        class_id = audio_info['class_id']
        if class_id not in class_total_segments:
            class_total_segments[class_id] = 0
        class_total_segments[class_id] += audio_info['max_segments']
    return class_total_segments

def extract_balanced_segments(audio_files, cap_per_class, segment_sec, sr, class_total_segments):
    """Extract segments with balanced sampling across classes."""
    segments = []
    class_counts = {}
    active_audios = audio_files.copy()
    segment_index = 0
    
    while active_audios:
        audios_to_remove = []
        
        for i, audio_info in enumerate(active_audios):
            class_id = audio_info['class_id']
            
            # Initialize class counter
            if class_id not in class_counts:
                class_counts[class_id] = 0
            
            # Check if class has reached cap or audio is exhausted
            if class_counts[class_id] >= cap_per_class:
                audios_to_remove.append(i)
                continue
                
            if segment_index >= audio_info['max_segments']:
                audios_to_remove.append(i)
                continue
            
            # Extract and validate segment
            segment_data = extract_single_segment(audio_info, segment_index)
            
            if segment_data is not None:
                segment_data['class_total_segments'] = class_total_segments[class_id]
                segments.append(segment_data)
                class_counts[class_id] += 1
        
        # Remove exhausted audios
        for i in sorted(audios_to_remove, reverse=True):
            active_audios.pop(i)
        
        segment_index += 1
        
        if not active_audios:
            break
    
    return segments

def extract_single_segment(audio_info, segment_index):
    """Extract a single segment from audio data if it passes RMS threshold."""
    start_sample = segment_index * audio_info['samples_per_segment']
    end_sample = start_sample + audio_info['samples_per_segment']
    segment = audio_info['audio_data'][start_sample:end_sample]
    
    # Check RMS threshold
    seg_rms = np.mean(lbrs.feature.rms(y=segment)[0])
    if seg_rms < audio_info['threshold']:
        return None
    
    return {
        'audio_data': segment,
        'class_id': audio_info['class_id'],
        'author': audio_info['author'],
        'original_filename': audio_info['filename'],
        'segment_index': segment_index,
        'sr': audio_info['sr']
    }

def create_single_spectrogram(segment_info, spectrogram_dir, mels, hoplen, nfft):
    """Create a single spectrogram from segment data."""
    try:
        # Generate spectrogram filename
        base_filename = os.path.splitext(segment_info['original_filename'])[0]
        segment_filename = f"{base_filename}_{segment_info['segment_index']}.wav"
        
        # Create spectrogram image
        img, spec_path, spec_name = get_spec_image(
            segment_info['audio_data'], 
            sr=segment_info['sr'], 
            mels=mels, 
            hoplen=hoplen, 
            nfft=nfft,
            filename=segment_filename, 
            start=segment_info['segment_index'], 
            spectrogram_dir=spectrogram_dir
        )
        
        # Save spectrogram image
        Image.fromarray(img).save(spec_path)
        
        return {
            'filename': spec_name,
            'class_id': segment_info['class_id'],
            'author': segment_info['author'],
            'species_segments': segment_info['class_total_segments']
        }
        
    except Exception as e:
        print(f"Error creating spectrogram for segment {segment_info['segment_index']} "
                f"of {segment_info['original_filename']}: {e}")
        return None

def save_test_audio(segment_info, test_audios_dir):
    """Save a segment as test audio file."""
    import soundfile as sf
    base_filename = os.path.splitext(segment_info['original_filename'])[0]
    test_filename = f"{base_filename}_{segment_info['segment_index']}_test.wav"
    test_path = os.path.join(test_audios_dir, test_filename)
    sf.write(test_path, segment_info['audio_data'], segment_info['sr'])

def plot_summary(final_df, output_csv_path):
    """Plot summary histogram of spectrogram generation showing samples per class."""
    if len(final_df) == 0:
        print("No spectrograms were created - empty DataFrame!")
        return
        
    if 'class_id' not in final_df.columns:
        print(f"Warning: 'class_id' column not found in DataFrame. Columns: {list(final_df.columns)}")
        return
        
    class_counts = final_df['class_id'].value_counts().sort_index()
    # Create histogram plot
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.index, class_counts.values, alpha=0.7)
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Samples per Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

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
                    num_epochs, device, fold_num=None, estop=25):
    """Train model on a single fold and return training history including F1 scores."""
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    best_epoch = 0
    early_stopped = False
    
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
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
        
        # Check if we should stop early
        if epochs_without_improvement >= estop and epoch > estop + 10:
            early_stopped = True
            break
        
        pbar.set_description(f"{desc} - Epoch {epoch+1}/{num_epochs}")
        pbar.set_postfix(
            train_acc=f"{train_acc:.3f}", 
            train_loss=f"{train_loss:.4f}",
            train_f1=f"{train_f1:.3f}",
            val_acc=f"{val_acc:.3f}",
            val_loss=f"{val_loss:.4f}",
            val_f1=f"{val_f1:.3f}",
            best_val_loss=f"{best_val_loss:.4f}",
            no_improve=epochs_without_improvement
        )
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # print(f"Restored model weights from epoch {best_epoch + 1} (best val loss: {best_val_loss:.4f})")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'early_stopped': early_stopped,
        'total_epochs': epoch + 1 if early_stopped else num_epochs
    }

def k_fold_cross_validation(dataset, model_class, num_classes, k_folds=4, 
                            num_epochs=300, batch_size=32, lr=0.001, random_state=435, 
                            aggregate_predictions=True, use_class_weights=True, estop=25,
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
                                If False, use mean of individual fold losses
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
            # Calculate standardization statistics from training data only
            train_data = torch.stack([dataset[i][0] for i in train_ids])
            train_mean = train_data.mean()
            train_std = train_data.std()
            
            print(f"Training data statistics - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
            
            # Create standardized datasets
            standardized_train_data = []
            standardized_val_data = []
            
            # Standardize training data
            for i in train_ids:
                x, y = dataset[i]
                x_standardized = (x - train_mean) / (train_std + 1e-8)  # Add small epsilon to avoid division by zero
                standardized_train_data.append((x_standardized, y))
            
            # Standardize validation data using training statistics
            for i in val_ids:
                x, y = dataset[i]
                x_standardized = (x - train_mean) / (train_std + 1e-8)
                standardized_val_data.append((x_standardized, y))
            
            # Create subsets with standardized data
            train_subset = standardized_train_data
            val_subset = standardized_val_data
        else:
            # Create data subsets without standardization
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
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=12,
                pin_memory=True,
                persistent_workers=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=12,
                pin_memory=True,
                persistent_workers=True
            )
        else:
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
        
        # Initialize model and optimizer
        model = model_class(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train the fold
        fold_history = train_single_fold(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs, device, fold_num=fold+1, estop=estop
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
            'model_state': model.state_dict().copy(),  # Save best model if needed
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
                                                    estop=25, standardize=False):
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
            # Calculate standardization statistics from training data only
            train_data = torch.stack([dataset[i][0] for i in train_ids])
            train_mean = train_data.mean()
            train_std = train_data.std()
            
            print(f"Training data statistics - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
            
            # Create standardized datasets
            standardized_train_data = []
            standardized_val_data = []
            
            # Standardize training data
            for i in train_ids:
                x, y = dataset[i]
                x_standardized = (x - train_mean) / (train_std + 1e-8)
                standardized_train_data.append((x_standardized, y))
            
            # Standardize validation data using training statistics
            for i in val_ids:
                x, y = dataset[i]
                x_standardized = (x - train_mean) / (train_std + 1e-8)
                standardized_val_data.append((x_standardized, y))
            
            # Create subsets with standardized data
            train_subset = standardized_train_data
            val_subset = standardized_val_data
        else:
            # Create data subsets without standardization
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
            criterion = nn.CrossEntropyLoss()        # Create data loaders
        if standardize:
            # For standardized data, we need custom DataLoader handling
            train_dataset = StandardizedDataset(train_subset)
            val_dataset = StandardizedDataset(val_subset)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=12,
                pin_memory=True,
                persistent_workers=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=12,
                pin_memory=True,
                persistent_workers=True
            )
        else:
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
        
        # Initialize model and optimizer
        model = model_class(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train the fold
        fold_history = train_single_fold(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs, device, fold_num=fold+1, estop=estop
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
        # Use mean of fold losses
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

def single_fold_training(dataset, model_class, num_classes, num_epochs=250, 
                        batch_size=48, lr=0.001, test_size=0.2, random_state=435, 
                        use_class_weights=True, estop=25):
    """
    Perform single fold training with 80-20 split and early stopping.
    
    Args:
        dataset: PyTorch dataset containing all data
        model_class: Model class to instantiate (e.g., models.BirdCNN)
        num_classes: Number of output classes
        num_epochs: Number of epochs to train
        batch_size: Batch size for data loaders
        lr: Learning rate
        test_size: Fraction of data to use for validation (0.2 = 20%)
        random_state: Random seed for reproducibility
        use_class_weights: If True, compute and use class weights for CrossEntropyLoss
        estop: Number of epochs without improvement before early stopping
    
    Returns:
        Dictionary containing training history and final model
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create train-validation split
    indices = list(range(len(dataset)))
    train_ids, val_ids = train_test_split(
        indices, test_size=test_size, random_state=random_state, 
        stratify=[dataset[i][1].item() for i in indices]  # stratify by labels
    )
    
    print(f"Training on {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}")
    
    # Create data subsets
    train_subset = Subset(dataset, train_ids)
    val_subset = Subset(dataset, val_ids)
    
    # Compute class weights if enabled
    if use_class_weights:
        train_labels = [dataset[i][1].item() for i in train_ids]
        unique_classes = np.unique(train_labels)
        
        class_weights_array = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=train_labels
        )
        
        class_weights = torch.ones(num_classes)
        for i, cls in enumerate(unique_classes):
            class_weights[cls] = class_weights_array[i]
        
        class_weights = class_weights.to(device)
        print(f"Class weights computed: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
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
    
    # Initialize model and optimizer
    model = model_class(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    history = train_single_fold(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs, device, fold_num=None, estop=estop
    )
    
    # Get final validation metrics
    final_val_loss, final_val_acc, final_val_f1 = validate_epoch(
        model, val_loader, criterion, device
    )
    
    results = {
        'history': history,
        'final_val_acc': final_val_acc,
        'final_val_loss': final_val_loss,
        'final_val_f1': final_val_f1,
        'best_val_acc': max(history['val_accuracies']),
        'best_val_f1': max(history['val_f1s']),
        'model': model,
        'model_state': model.state_dict().copy(),
        'config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'test_size': test_size,
            'device': str(device),
            'use_class_weights': use_class_weights,
            'estop': estop
        }
    }
    
    print(f"\nTraining Complete!")
    if history['early_stopped']:
        print(f"Early stopped after {history['total_epochs']} epochs (best at epoch {history['best_epoch'] + 1})")
    print(f"Final - Val Acc: {final_val_acc:.4f}, Val Loss: {final_val_loss:.4f}, Val F1: {final_val_f1:.4f}")
    print(f"Best - Val Acc: {results['best_val_acc']:.4f}, Val F1: {results['best_val_f1']:.4f}")
    
    return results


# Display Results
def plot_best_results(best_results, metric_key, title, ylabel, ax=None):
    ''' Given a dictionary of lists with the best results for each fold in 
    each of the metric keys, plot a bar graph showing the best results for each
    fold in the given metric key, and the average of all the folds '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
    
    # Get the values for the specified metric
    values = best_results[metric_key]
    
    # Calculate the average
    avg_value = np.mean(values)
    
    # Create fold labels
    fold_labels = [f'Fold {i+1}' for i in range(len(values))]
    fold_labels.append('Average')
    
    # Add average to values for plotting
    plot_values = values + [avg_value]
    
    # Create the bar plot
    bars = ax.bar(fold_labels, plot_values, alpha=0.7)
    
    # Color the average bar differently
    bars[-1].set_color('red')
    bars[-1].set_alpha(0.8)
    
    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, plot_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(plot_values)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold' if i == len(bars)-1 else 'normal')
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Fold')
    ax.grid(True, alpha=0.3)
    
    if show_plot:
        plt.tight_layout()
        plt.show()

def plot_mean_curve(results, metric_key, title, ylabel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        show_plot = True
    else:
        show_plot = False
    
    all_train = []
    all_val = []
    max_epochs = 0

    # First pass: collect all curves and find max epochs
    for fold_data in results['fold_results'].values():
        history = fold_data['history']
        train_curve = history[f"train_{metric_key}"]
        val_curve = history[f"val_{metric_key}"]
        all_train.append(train_curve)
        all_val.append(val_curve)
        max_epochs = max(max_epochs, len(train_curve))

    # Pad shorter curves with NaN to align all curves to max_epochs
    for i in range(len(all_train)):
        current_length = len(all_train[i])
        if current_length < max_epochs:
            # Pad with NaN values
            all_train[i] = all_train[i] + [np.nan] * (max_epochs - current_length)
            all_val[i] = all_val[i] + [np.nan] * (max_epochs - current_length)

    # Convert to arrays for averaging (nanmean will ignore NaN values)
    all_train = np.array(all_train)
    all_val = np.array(all_val)

    mean_train = np.nanmean(all_train, axis=0)
    mean_val = np.nanmean(all_val, axis=0)

    ax.plot(mean_train, label="Mean Train", linestyle='--')
    ax.plot(mean_val, label="Mean Val")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    if show_plot:
        plt.tight_layout()
        plt.show()

def plot_kfold_results(results, best_results):
    """
    Plot comprehensive K-fold cross validation results in a 3x2 grid.
    Left column shows mean curves over epochs, right column shows best results per fold.
    
    Args:
        results: Dictionary containing k-fold cross validation results
        best_results: Dictionary containing best results for each fold
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("K-Fold Cross Validation Results", fontsize=16, y=0.98)
    
    # Row 1: Accuracies
    plot_mean_curve(results, "accuracies", "Accuracy Curves Across Folds", "Accuracy", ax=axes[0, 0])
    plot_best_results(best_results, "accuracies", "Best Accuracy per Fold", "Accuracy", ax=axes[0, 1])
    
    # Row 2: F1 Scores
    plot_mean_curve(results, "f1s", "F1 Score Curves Across Folds", "Macro F1 Score", ax=axes[1, 0])
    plot_best_results(best_results, "f1s", "Best F1 Score per Fold", "F1 Score", ax=axes[1, 1])
    
    # Row 3: Losses
    plot_mean_curve(results, "losses", "Loss Curves Across Folds", "Cross Entropy Loss", ax=axes[2, 0])
    plot_best_results(best_results, "losses", "Best Loss per Fold", "Loss", ax=axes[2, 1])
    
    plt.tight_layout()
    plt.show()

def plot_single_fold_curve(results, metric_key, title, ylabel):
    """
    Plot a single training curve for single fold training results.
    
    Args:
        results: Dictionary containing training history from single_fold_training
        metric_key: Key for the metric in history (e.g., 'accuracies', 'losses', 'f1s')
        title: Title for the plot
        ylabel: Label for y-axis
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['history'][f'train_{metric_key}'], label=f'Train {ylabel}', linestyle='--')
    plt.plot(results['history'][f'val_{metric_key}'], label=f'Val {ylabel}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def print_single_fold_results(results):
    """
    Print formatted results from single fold training.
    
    Args:
        results: Dictionary containing training results from single_fold_training
    """
    print(f"Final Validation Accuracy: {results['final_val_acc']:.4f}")
    print(f"Final Validation F1 Score: {results['final_val_f1']:.4f}")
    print(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
    print(f"Best Validation F1 Score: {results['best_val_f1']:.4f}")

def print_kfold_best_results(results):
    """
    Print best results for each metric from k-fold cross validation alongside the epoch they occurred.
    
    Args:
        results: Dictionary containing k-fold cross validation results
    """
    print("K-Fold Cross Validation - Best Results per Fold:")
    print("="*60)
    
    for fold_name, fold_data in results['fold_results'].items():
        print(f"\n{fold_name.upper()}:")
        
        # Find best epochs for each metric
        best_val_acc_epoch = fold_data['val_accuracies'].index(fold_data['best_val_acc']) + 1
        best_val_f1_epoch = fold_data['val_f1s'].index(fold_data['best_val_f1']) + 1
        best_val_loss_epoch = fold_data['val_losses'].index(min(fold_data['val_losses'])) + 1
        
        print(f"  Best Val Accuracy: {fold_data['best_val_acc']:.4f} (Epoch {best_val_acc_epoch})")
        print(f"  Best Val F1 Score: {fold_data['best_val_f1']:.4f} (Epoch {best_val_f1_epoch})")
        print(f"  Best Val Loss: {min(fold_data['val_losses']):.4f} (Epoch {best_val_loss_epoch})")
    
    print(f"\nOVERALL SUMMARY:")
    print(f"Mean Val Accuracy: {results['summary']['mean_val_acc']:.4f} ± {results['summary']['std_val_acc']:.4f}")
    print(f"Mean Val F1 Score: {results['summary']['mean_val_f1']:.4f} ± {results['summary']['std_val_f1']:.4f}")
    
    if 'aggregated_f1' in results['summary']:
        print(f"Aggregated F1 Score: {results['summary']['aggregated_f1']:.4f}")


# Model Utils
def save_model(model, model_name, model_save_path):
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to: {model_save_path}")

def test_saved_model(save_path):
    state = torch.load(save_path, map_location='cpu')
    print(type(state))
    print(list(state.keys())[:5])  # show first 5 parameter names
    print(state[list(state.keys())[0]].shape)  # show shape of first tensor

def load_model(model_class, model_name, num_classes=29):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=num_classes).to(device)
    model_path = os.path.join('..', 'models', f"{model_name}.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def reset_model(model_class, lr=0.001, num_classes=29):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion, device



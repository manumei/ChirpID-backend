# Data Processing Utilities
# Handles audio loading, segmentation, spectrogram creation, and file I/O operations

import os
import shutil
import numpy as np
import librosa
import soundfile as sf
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import noisereduce as nr

# Directory Utilities
def clean_dir(dest_dir):
    """Deletes the raw audio files in the dest_dir."""
    print(f"Resetting {dest_dir} directory...")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

def count_files_in_dir(dir_path):
    """Counts the number of files in a directory."""
    if not os.path.exists(dir_path):
        return 0
    return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

# Audio Loading and Processing
def lbrs_loading(audio_path, sr, mono=True):
    """Load audio file using librosa with sample rate validation."""
    y, srate = librosa.load(audio_path, sr=sr, mono=mono)
    if srate != sr:
        raise ValueError(f"Sample rate mismatch: expected {sr}, got {srate}, at audio file {audio_path}")
    return y, srate

def get_rmsThreshold(y, frame_len, hop_len, thresh_factor=0.5):
    """Calculate RMS energy threshold for audio segmentation."""
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    threshold = thresh_factor * np.mean(rms)
    return threshold

def reduce_noise_seg(segment, srate, filename, class_id):
    """Apply noise reduction to audio segment."""
    try:
        segment = nr.reduce_noise(y=segment, sr=srate, stationary=False)
    except RuntimeWarning as e:
        print(f"RuntimeWarning while reducing noise for segment in {filename} from {class_id}: {e}")
    except Exception as e:
        print(f"Error while reducing noise for segment in {filename} from {class_id}: {e}")
    return segment

# Spectrogram Processing
def get_spec_norm(segment, sr, mels, hoplen, nfft):
    """Generate normalized mel spectrogram."""
    spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=mels, hop_length=hoplen, n_fft=nfft)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    
    # Check for invalid values or zero range
    if np.any(np.isnan(spec_db)) or np.any(np.isinf(spec_db)):
        raise ValueError("Spectrogram contains NaN or infinite values")
    
    spec_min = spec_db.min()
    spec_max = spec_db.max()
    spec_range = spec_max - spec_min
    
    # Check for zero range (all values are the same)
    if spec_range == 0 or spec_range < 1e-8:
        raise ValueError("Spectrogram has zero or near-zero dynamic range")
    
    norm_spec = (spec_db - spec_min) / spec_range
    
    # Final validation
    if np.any(np.isnan(norm_spec)) or np.any(np.isinf(norm_spec)):
        raise ValueError("Normalized spectrogram contains invalid values")
    
    return norm_spec

def get_spec_image(segment, sr, mels, hoplen, nfft, filename, start, spectrogram_dir):
    """Create and save spectrogram image."""
    norm_spec = get_spec_norm(segment, sr, mels, hoplen, nfft)
    
    # Ensure values are in valid range [0, 1] before scaling
    norm_spec = np.clip(norm_spec, 0.0, 1.0)
    
    # Convert to uint8 safely
    img = (norm_spec * 255).astype(np.uint8)
    
    spec_filename = f"{os.path.splitext(filename)[0]}_{start}.png"
    spec_path = os.path.join(spectrogram_dir, spec_filename)
    return img, spec_path, spec_filename

def get_spec_npy(segment, sr, mels, hoplen, nfft, filename, start, spectrogram_dir):
    """Create and save spectrogram as .npy file with float32 precision."""
    norm_spec = get_spec_norm(segment, sr, mels, hoplen, nfft)
    
    # Ensure values are in valid range [0, 1] and maintain float32 precision
    norm_spec = np.clip(norm_spec, 0.0, 1.0).astype(np.float32)
    
    spec_filename = f"{os.path.splitext(filename)[0]}_{start}.npy"
    spec_path = os.path.join(spectrogram_dir, spec_filename)
    return norm_spec, spec_path, spec_filename

def save_test_audios(segment, sr, test_audios_dir, filename, start, saved_audios):
    """Save test audio segments."""
    if test_audios_dir is not None and saved_audios < 10:
        import soundfile as sf
        os.makedirs(test_audios_dir, exist_ok=True)
        test_audio_filename = f"{os.path.splitext(filename)[0]}_{start}_test.wav"
        test_audio_path = os.path.join(test_audios_dir, test_audio_filename)
        sf.write(test_audio_path, segment, sr)

# Audio Segment Management
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
    Load audio segments from disk using the CSV metadata file.
    
    Args:
        segments_csv_path (str): Path to the CSV file with segment metadata
        segments_dir (str): Directory containing the audio segment files
        sr (int): Expected sample rate
        
    Returns:
        list: List of segment dictionaries ready for processing
    """
    import soundfile as sf
    
    segments_df = pd.read_csv(segments_csv_path)
    segments = []
    
    # print(f"Loading {len(segments_df)} audio segments from {segments_dir}")
    
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
    
    # print(f"Loaded {len(segments)} audio segments from disk")
    return segments

# Audio File Loading and Segmentation
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
    """Calculate total potential segments per class."""
    class_totals = {}
    
    for audio_info in audio_files:
        class_id = audio_info['class_id']
        segments = audio_info['max_segments']
        
        if class_id not in class_totals:
            class_totals[class_id] = 0
        class_totals[class_id] += segments
    
    return class_totals

def extract_balanced_segments(audio_files, cap_per_class, segment_sec, sr, class_total_segments):
    """Extract balanced segments from audio files."""
    class_segments_extracted = {class_id: 0 for class_id in class_total_segments.keys()}
    all_segments = []
    
    for audio_info in tqdm(audio_files, desc="Extracting segments"):
        class_id = audio_info['class_id']
        
        if class_segments_extracted[class_id] >= cap_per_class:
            continue
        
        y = audio_info['audio']
        threshold = audio_info['threshold']
        filename = audio_info['filename']
        author = audio_info['author']
        
        segment_samples = int(segment_sec * sr)
        
        for start_idx in range(0, len(y) - segment_samples + 1, segment_samples):
            if class_segments_extracted[class_id] >= cap_per_class:
                break
            
            segment = y[start_idx:start_idx + segment_samples]
            
            # Check if segment has enough energy
            rms = np.sqrt(np.mean(segment**2))
            if rms > threshold:
                all_segments.append({
                    'filename': filename,
                    'class_id': class_id,
                    'author': author,
                    'segment': segment,
                    'segment_idx': len(all_segments),
                    'sr': sr
                })
                class_segments_extracted[class_id] += 1
    
    return all_segments

def extract_single_segment(audio_info, segment_index):
    """Extract a single segment from audio info."""
    samples_per_segment = audio_info['samples_per_segment']
    start_sample = segment_index * samples_per_segment
    end_sample = start_sample + samples_per_segment
    
    # Extract segment
    segment_audio = audio_info['audio_data'][start_sample:end_sample]
    
    # Check RMS threshold
    seg_rms = np.mean(librosa.feature.rms(y=segment_audio)[0])
    
    if seg_rms < audio_info['threshold']:
        return None  # Skip low-energy segments
    
    return {
        'audio_data': segment_audio,
        'class_id': audio_info['class_id'],
        'author': audio_info['author'],
        'original_filename': audio_info['filename'],
        'segment_index': segment_index,
        'sr': audio_info['sr'],
        'class_total_segments': None  # To be filled later
    }

def create_single_spectrogram(segment_info, spectrogram_dir, mels, hoplen, nfft):
    """Create a single spectrogram from segment info."""
    audio_name = f"{segment_info['original_filename']}_segment_{segment_info['segment_index']}"
    
    try:
        # Validate audio data first
        if segment_info['audio_data'] is None or len(segment_info['audio_data']) == 0:
            raise ValueError("Empty or invalid audio data")
        
        # Check for silent or near-silent audio
        if np.max(np.abs(segment_info['audio_data'])) < 1e-8:
            raise ValueError("Audio segment is silent or has extremely low amplitude")
        
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
        print(f"{audio_name} has been removed due to {e} error")
        return None

def create_single_spectrogram_npy(segment_info, spectrogram_dir, mels, hoplen, nfft):
    """Create a single spectrogram from segment info and save as .npy file."""
    audio_name = f"{segment_info['original_filename']}_segment_{segment_info['segment_index']}"
    
    try:
        # Validate audio data first
        if segment_info['audio_data'] is None or len(segment_info['audio_data']) == 0:
            raise ValueError("Empty or invalid audio data")
        
        # Check for silent or near-silent audio
        if np.max(np.abs(segment_info['audio_data'])) < 1e-8:
            raise ValueError("Audio segment is silent or has extremely low amplitude")
        
        # Generate spectrogram filename
        base_filename = os.path.splitext(segment_info['original_filename'])[0]
        segment_filename = f"{base_filename}_{segment_info['segment_index']}.wav"
        
        # Create spectrogram as numpy array
        spec_array, spec_path, spec_name = get_spec_npy(
            segment_info['audio_data'], 
            sr=segment_info['sr'], 
            mels=mels, 
            hoplen=hoplen, 
            nfft=nfft,
            filename=segment_filename, 
            start=segment_info['segment_index'], 
            spectrogram_dir=spectrogram_dir
        )
        
        # Save spectrogram as .npy file
        np.save(spec_path, spec_array)
        
        return {
            'filename': spec_name,
            'class_id': segment_info['class_id'],
            'author': segment_info['author'],
            'species_segments': segment_info['class_total_segments']
        }
        
    except Exception as e:
        print(f"{audio_name} has been removed due to {e} error")
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

# Matrix Processing
def get_spect_matrix(image_path):
    """Get spectrogram matrix from image file."""
    img = Image.open(image_path).convert('L')
    pixels = np.array(img)
    return pixels

def get_spec_matrix_direct(segment, sr, mels, hoplen, nfft):
    """Get spectrogram matrix directly from segment and params."""
    norm_spec = get_spec_norm(segment, sr, mels, hoplen, nfft)
    matrix = np.clip(norm_spec, 0.0, 1.0).astype(np.float32)
    return matrix

def audio_process(audio_path, reduce_noise: bool, sr=32000, segment_sec=5.0,
                frame_len=2048, hop_len=512, mels=224, nfft=2048, thresh=0.75):
    """
    Takes the path to an audio file (any format) and processes it to finally return 
    the list of grayscale spectrogram pixel matrices for each of its high-RMS segments.

    Step 1: Load the audio file with librosa. (using lbrs_loading)
    Step 2: Split into high-RMS segments of 5 seconds. (using get_rmsThreshold)
    Step 3: Reduce noise for each segment if reduce_noise is True. (using reduce_noise_seg)
    Step 4: Generate a Spectrogram grayscale matrix for each segment. (using get_spec_matrix_direct)
    """
    matrices = []
    # print(f"Processing audio file: {audio_path}")
    samples_per_segment = int(sr * segment_sec)

    # Step 1
    y, srate = lbrs_loading(audio_path, sr)

    # Step 2
    threshold = get_rmsThreshold(y, frame_len, hop_len, thresh_factor=thresh)

    for start in range(0, len(y) - samples_per_segment + 1, samples_per_segment):
        segment = y[start:start + samples_per_segment]
        seg_rms = np.mean(librosa.feature.rms(y=segment)[0])
        
        if seg_rms < threshold:
            # print(f"Segment at {start} has {seg_rms} RMS, below threshold {threshold}. Skipping...")
            continue

        # Step 4
        filename = os.path.basename(audio_path)
        matrix = get_spec_matrix_direct(segment, srate, mels, hop_len, nfft)
        matrices.append(matrix)
    
    # print(f"Processed {len(matrices)} segments from {audio_path}")
    return matrices

if __name__ == "__main__":
    # Example usage
    audio_path = "database/audio/dev/XC112710.ogg"
    matrices = audio_process(audio_path, reduce_noise=True)
    
    # Preview first matrix
    if matrices:
        print(f"First matrix shape: {matrices[0].shape}")
        plt.imshow(matrices[0], cmap='gray')
        plt.title("Spectrogram Matrix")
        plt.axis('off')
        plt.show()
    
    # preview pixels
    if matrices:
        print(f"First matrix pixel values:\n{matrices[0]}")
        print(f"Matrix dtype: {matrices[0].dtype}, shape: {matrices[0].shape}")
    else:
        print("No valid segments found.")
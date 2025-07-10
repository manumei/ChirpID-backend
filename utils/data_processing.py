# Data Processing Utilities
# Handles audio loading, segmentation, spectrogram creation, and file I/O operations

import os
import shutil
import numpy as np
import librosa
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from utils.data_extraction import lbrs_loading

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
        base_filename = os.path.splitext(segment_info['filename'])[0]
        segment_filename = f"{base_filename}_{segment_info['segment_index']}.wav"
        segment_path = os.path.join(segments_output_dir, segment_filename)
        
        # Save audio segment to disk
        sf.write(segment_path, segment_info['segment'], segment_info['sr'])
        og_filename = segment_info['filename']
        og_audio_name = os.path.splitext(og_filename)[0] if og_filename else 'unknown'
        
        # Create record for CSV
        segment_records.append({
            'filename': segment_filename,
            'class_id': segment_info['class_id'],
            'author': segment_info['author'],
            'original_audio': og_audio_name,
            'segment_index': segment_info['segment_index'],
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
            }
            
            segments.append(segment_info)
            
        except Exception as e:
            print(f"Error loading segment {row['filename']}: {e}")
            continue
    
    # print(f"Loaded {len(segments)} audio segments from disk")
    return segments

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

# Spectrogram Management
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

def get_audio_info(audio_path, sr, length):
    y, srate = lbrs_loading(audio_path, sr)
    audio_duration = len(y) / srate
    
    # Check Duration is enough
    buffer = length * 1.005
    if audio_duration < buffer:
        raise ValueError(f"Audio is too short: {audio_duration:.2f}s. Minimum required: {length + 0.05:.2f}s")
    else:
        return y, srate

def audio_process(audio_path, sr=32000, segment_sec=5.0,
                frame_len=2048, hop_len=512, mels=224, nfft=2048, thresh=0.7):
    """
    Takes the path to an audio file (any format) and processes it to finally return 
    the list of grayscale spectrogram pixel matrices for each of its high-RMS segments.

    Step 1: Load the audio file with librosa. (using lbrs_loading)
    Step 2: Split into high-RMS segments of 5 seconds. (using get_rmsThreshold)
    Step 3: Reduce noise for each segment if reduce_noise is True. (using reduce_noise_seg)
    Step 4: Generate a Spectrogram grayscale matrix for each segment. (using get_spec_matrix_direct)
    """
    matrices = []
    samples_per = int(sr * segment_sec)
    
    y, srate = get_audio_info(audio_path, sr, segment_sec)
    
    # TODO: AUN QUEDA DEFINIR QUE TYPE DE THRESHOLD USAR (VER data_extraction.py)
    threshold = get_rmsThreshold(y, frame_len, hop_len, thresh_factor=thresh)
    
    for start in range(0, len(y) - samples_per + 1, samples_per):
        segment = y[start:start + samples_per]
        seg_rms = np.mean(librosa.feature.rms(y=segment)[0])
        
        if seg_rms < threshold:
            continue
        
        matrix = get_spec_matrix_direct(segment, srate, mels, hop_len, nfft)
        matrices.append(matrix)
    
    return matrices

if __name__ == "__main__":
    # Example usage
    audio_path = "database/audio/dev/XC112710.ogg"
    matrices = audio_process(audio_path)
    
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

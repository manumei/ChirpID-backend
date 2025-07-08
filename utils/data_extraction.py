import os
import numpy as np
import librosa
import noisereduce as nr

from tqdm import tqdm
from scipy.stats import entropy
from scipy.signal import find_peaks, butter, filtfilt

# Audio Loading and Processing
def lbrs_loading(audio_path, sr, mono=True):
    """Load audio file using librosa with sample rate validation."""
    y, srate = librosa.load(audio_path, sr=sr, mono=mono)
    if srate != sr:
        raise ValueError(f"Sample rate mismatch: expected {sr}, got {srate}, at audio file {audio_path}")
    return y, srate

def reduce_noise_seg(segment, srate, filename, class_id):
    """Apply noise reduction to audio segment."""
    try:
        segment = nr.reduce_noise(y=segment, sr=srate, stationary=False)
    except RuntimeWarning as e:
        print(f"RuntimeWarning while reducing noise for segment in {filename} from {class_id}: {e}")
    except Exception as e:
        print(f"Error while reducing noise for segment in {filename} from {class_id}: {e}")
    return segment


# Energy RMS Cut-Off
def get_rmsThreshold(y, frame_len, hop_len, thresh_factor=0.5):
    """Calculate RMS energy threshold for audio segmentation."""
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    threshold = thresh_factor * np.mean(rms)
    return threshold


# Energy Peaks Cut-Off
def get_peakThreshold(y, frame_len, hop_len, thresh_factor=0.5, percentile=85):
    """Calculate energy peak threshold for audio segmentation."""
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    # Use percentile instead of mean to capture peak behavior
    threshold = thresh_factor * np.percentile(rms, percentile)
    return threshold

def segment_has_energy_peaks(segment, threshold, sr, min_peak_height_ratio=0.8, 
                            min_peak_distance=0.1, prominence_factor=0.3):
    """
    Check if segment has significant energy peaks indicating bird activity.
    
    Args:
        segment: Audio segment
        threshold: Global peak threshold
        sr: Sample rate
        min_peak_height_ratio: Minimum peak height as ratio of threshold
        min_peak_distance: Minimum distance between peaks (seconds)
        prominence_factor: Minimum prominence as ratio of threshold
    """
    # Calculate RMS energy over time
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find peaks in RMS energy
    min_height = threshold * min_peak_height_ratio
    min_distance_frames = int(min_peak_distance * sr / hop_length)
    min_prominence = threshold * prominence_factor
    
    peaks, properties = find_peaks(
        rms, 
        height=min_height,
        distance=min_distance_frames,
        prominence=min_prominence
    )
    
    # Additional criteria
    if len(peaks) == 0:
        return False
    
    # Check if peaks are significant enough
    peak_heights = properties['peak_heights']
    max_peak = np.max(peak_heights)
    
    # Must have at least one strong peak OR multiple moderate peaks
    strong_peak_condition = max_peak >= threshold * 1.2
    multiple_peaks_condition = len(peaks) >= 2 and np.mean(peak_heights) >= threshold * 0.9
    
    return strong_peak_condition or multiple_peaks_condition


# Spectral Entropy Cut-Off
def calculate_spectral_entropy(y, sr, n_fft=2048, hop_length=512):
    """Calculate spectral entropy of audio signal."""
    # Compute magnitude spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Calculate power spectral density
    psd = magnitude ** 2
    
    # Normalize each frame to create probability distribution
    psd_normalized = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-10)
    
    # Calculate entropy for each frame
    spectral_entropies = []
    for frame in psd_normalized.T:
        # Remove zeros to avoid log(0)
        frame_nonzero = frame[frame > 1e-10]
        if len(frame_nonzero) > 0:
            frame_entropy = entropy(frame_nonzero, base=2)
            spectral_entropies.append(frame_entropy)
        else:
            spectral_entropies.append(0.0)
    
    return np.array(spectral_entropies)

def get_spectralThreshold(y, sr, frame_len, hop_len, thresh_factor=0.5, entropy_weight=0.6):
    """Calculate combined spectral entropy and flatness threshold."""
    # Calculate spectral entropy
    spectral_entropies = calculate_spectral_entropy(y, sr, n_fft=frame_len, hop_length=hop_len)
    
    # Calculate spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame_len, hop_length=hop_len)[0]
    
    # Combine metrics (higher entropy and lower flatness indicate more complex/interesting audio)
    # Invert flatness so higher values indicate more activity
    inverted_flatness = 1.0 - spectral_flatness
    
    # Weighted combination
    combined_metric = (entropy_weight * spectral_entropies + 
                      (1 - entropy_weight) * inverted_flatness)
    
    # Set threshold based on percentile to capture active segments
    threshold = thresh_factor * np.percentile(combined_metric, 70)
    
    return threshold

def segment_has_spectral_complexity(segment, threshold, sr, entropy_weight=0.6, 
                                    min_active_frames=0.3):
    """
    Check if segment has sufficient spectral complexity indicating bird activity.
    
    Args:
        segment: Audio segment
        threshold: Global spectral threshold
        sr: Sample rate
        entropy_weight: Weight for entropy vs flatness
        min_active_frames: Minimum fraction of frames that must be above threshold
    """
    # Calculate spectral metrics for the segment
    spectral_entropies = calculate_spectral_entropy(segment, sr, n_fft=2048, hop_length=512)
    spectral_flatness = librosa.feature.spectral_flatness(y=segment, n_fft=2048, hop_length=512)[0]
    
    # Combine metrics
    inverted_flatness = 1.0 - spectral_flatness
    
    # Ensure arrays are same length
    min_len = min(len(spectral_entropies), len(inverted_flatness))
    spectral_entropies = spectral_entropies[:min_len]
    inverted_flatness = inverted_flatness[:min_len]
    
    combined_metric = (entropy_weight * spectral_entropies + 
                      (1 - entropy_weight) * inverted_flatness)
    
    # Check what fraction of frames are above threshold
    active_frames = np.sum(combined_metric > threshold)
    active_fraction = active_frames / len(combined_metric)
    
    # Additional checks
    mean_complexity = np.mean(combined_metric)
    max_complexity = np.max(combined_metric)
    
    # Pass if segment has sufficient spectral activity
    frame_activity_condition = active_fraction >= min_active_frames
    mean_activity_condition = mean_complexity >= threshold * 0.8
    peak_activity_condition = max_complexity >= threshold * 1.5
    
    return frame_activity_condition or (mean_activity_condition and peak_activity_condition)


# Band-Pass Filter Cut-Off
def create_bird_bandpass_filter(sr, low_freq=1000, high_freq=8000, order=5):
    """Create a bandpass filter optimized for bird vocalizations."""
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are within valid range
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(y, sr, low_freq=1000, high_freq=8000, order=5):
    """Apply bandpass filter to audio signal."""
    b, a = create_bird_bandpass_filter(sr, low_freq, high_freq, order)
    
    # Apply zero-phase filter to avoid phase distortion
    filtered_y = filtfilt(b, a, y)
    return filtered_y

def get_bandpassThreshold(y, sr, frame_len, hop_len, thresh_factor=0.5, 
                        low_freq=1000, high_freq=8000):
    """Calculate RMS threshold using bandpass-filtered audio for detection."""
    # Apply bandpass filter for threshold calculation
    filtered_y = apply_bandpass_filter(y, sr, low_freq, high_freq)
    
    # Calculate RMS on filtered signal
    rms = librosa.feature.rms(y=filtered_y, frame_length=frame_len, hop_length=hop_len)[0]
    threshold = thresh_factor * np.mean(rms)
    
    return threshold

def segment_has_bandpass_activity(segment, threshold, sr, low_freq=1000, high_freq=8000,
                                energy_ratio_threshold=0.7, min_duration_ratio=0.2):
    """
    Check if segment has bird activity using bandpass filtering for detection only.
    
    Args:
        segment: Original audio segment (unfiltered)
        threshold: Global bandpass threshold
        sr: Sample rate
        low_freq: Low cutoff frequency for bird detection
        high_freq: High cutoff frequency for bird detection
        energy_ratio_threshold: Minimum ratio of bandpass energy to total energy
        min_duration_ratio: Minimum fraction of segment that must be active
    """
    # Apply bandpass filter ONLY for detection
    filtered_segment = apply_bandpass_filter(segment, sr, low_freq, high_freq)
    
    # Calculate RMS energy on filtered signal
    filtered_rms = np.sqrt(np.mean(filtered_segment**2))
    
    # Primary check: filtered energy above threshold
    basic_threshold_check = filtered_rms > threshold
    
    # Additional checks for robustness
    # 1. Energy ratio check: bird frequencies should contain significant energy
    original_energy = np.mean(segment**2)
    filtered_energy = np.mean(filtered_segment**2)
    
    if original_energy > 1e-10:  # Avoid division by zero
        energy_ratio = filtered_energy / original_energy
        energy_ratio_check = energy_ratio > energy_ratio_threshold
    else:
        energy_ratio_check = False
    
    # 2. Temporal activity check: look for sustained activity in bird frequency range
    frame_length = 2048
    hop_length = 512
    filtered_rms_frames = librosa.feature.rms(y=filtered_segment, 
                                            frame_length=frame_length, 
                                            hop_length=hop_length)[0]
    
    active_frames = np.sum(filtered_rms_frames > threshold * 0.8)
    total_frames = len(filtered_rms_frames)
    activity_ratio = active_frames / total_frames if total_frames > 0 else 0
    
    temporal_activity_check = activity_ratio > min_duration_ratio
    
    # 3. Peak activity check: look for significant peaks in bird frequency range
    max_filtered_rms = np.max(filtered_rms_frames)
    peak_activity_check = max_filtered_rms > threshold * 1.2
    
    # Combine checks: basic threshold AND (energy ratio OR temporal activity OR peak activity)
    return basic_threshold_check and (energy_ratio_check or temporal_activity_check or peak_activity_check)


# Final Extraction Functions
def load_audio_files(segments_df, segments_dir, sr, segment_sec, thresh_factor, cutoff_type="peak"):
    """
    Load and prepare audio files with metadata for segment extraction.
    
    Args:
        segments_df: DataFrame containing audio file metadata
        segments_dir: Directory path containing audio files
        sr: Sample rate for audio loading
        segment_sec: Duration of each segment in seconds
        thresh_factor: Threshold factor for cutoff calculations
        cutoff_type: Method for threshold calculation - "rms", "peak", "entropy", "filter", or "uncut"
    
    Returns:
        List of dictionaries containing audio data and metadata for segment extraction
    """
    
    audio_files = []
    samples_per_segment = int(sr * segment_sec)
    
    for _, row in segments_df.iterrows():
        filename = row['filename']
        class_id = row['class_id']
        author = row['author']
        audio_path = os.path.join(segments_dir, filename)
        
        try:
            y, srate = lbrs_loading(audio_path, sr=sr, mono=True)
            
            # Calculate threshold based on cutoff type
            if cutoff_type == "rms":
                threshold = get_rmsThreshold(y, frame_len=2048, hop_len=512, thresh_factor=thresh_factor)
            elif cutoff_type == "peak":
                threshold = get_peakThreshold(y, frame_len=2048, hop_len=512, 
                                            thresh_factor=thresh_factor, percentile=85)
            elif cutoff_type == "entropy":
                threshold = get_spectralThreshold(y, srate, frame_len=2048, hop_length=512, 
                                                thresh_factor=thresh_factor)
            elif cutoff_type == "filter":
                threshold = get_bandpassThreshold(y, srate, frame_len=2048, hop_length=512, 
                                                thresh_factor=thresh_factor)
            elif cutoff_type == "uncut":
                threshold = 0.0  # No threshold for uncut segments
            else:
                raise ValueError(f"Invalid cutoff_type: {cutoff_type}. Must be one of: 'rms', 'peak', 'entropy', 'filter', 'uncut'")
            
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

def extract_balanced_segments(audio_files, cap_per_class, segment_sec, sr, class_total_segments, cutoff_type="peak"):
    """
    Extract balanced segments from audio files using specified cutoff method.
    
    Args:
        audio_files: List of dictionaries containing audio data and metadata
        cap_per_class: Maximum number of segments to extract per class
        segment_sec: Duration of each segment in seconds
        sr: Sample rate
        class_total_segments: Dictionary mapping class IDs to total available segments
        cutoff_type: Method for segment validation - "rms", "peak", "entropy", "filter", or "uncut"
    
    Returns:
        List of dictionaries containing extracted segments with metadata
    """
    
    class_segments_extracted = {class_id: 0 for class_id in class_total_segments.keys()}
    all_segments = []
    
    desc_map = {
        "rms": "Extracting segments (RMS Energy)",
        "peak": "Extracting segments (Energy Peaks)",
        "entropy": "Extracting segments (Spectral Analysis)",
        "filter": "Extracting segments (Bandpass Filter)",
        "uncut": "Extracting segments (No Cutoffs)"
    }
    
    for audio_info in tqdm(audio_files, desc=desc_map.get(cutoff_type, "Extracting segments")):
        class_id = audio_info['class_id']
        
        if class_segments_extracted[class_id] >= cap_per_class:
            continue
        
        y = audio_info['audio_data']
        threshold = audio_info['threshold']
        filename = audio_info['filename']
        author = audio_info['author']
        
        segment_samples = int(segment_sec * sr)
        
        for start_idx in range(0, len(y) - segment_samples + 1, segment_samples):
            if class_segments_extracted[class_id] >= cap_per_class:
                break
            
            segment = y[start_idx:start_idx + segment_samples]
            
            # Check segment validity based on cutoff type
            is_valid = False
            
            if cutoff_type == "rms":
                rms = np.sqrt(np.mean(segment**2))
                is_valid = rms > threshold
            elif cutoff_type == "peak":
                is_valid = segment_has_energy_peaks(segment, threshold, sr)
            elif cutoff_type == "entropy":
                is_valid = segment_has_spectral_complexity(segment, threshold, sr)
            elif cutoff_type == "filter":
                is_valid = segment_has_bandpass_activity(segment, threshold, sr)
            elif cutoff_type == "uncut":
                is_valid = True  # Accept all segments for uncut
            else:
                raise ValueError(f"Invalid cutoff_type: {cutoff_type}. Must be one of: 'rms', 'peak', 'entropy', 'filter', 'uncut'")
            
            if is_valid:
                all_segments.append({
                    'filename': filename,
                    'class_id': class_id,
                    'author': author,
                    'segment': segment,
                    'segment_index': len(all_segments),
                    'sr': sr
                })
                class_segments_extracted[class_id] += 1
    
    return all_segments

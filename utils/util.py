# Directory Aux
import os
import shutil

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
                frame_len=2048, hop_len=512, mels=128, nfft=2048):
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
    threshold = get_rmsThreshold(y, frame_len, hop_len)

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

# ChirpID Backend - Spectrogram Pipeline Conversion Summary

## Changes Made

### 1. New Functions Added to `utils/data_processing.py`:

#### `get_spec_npy(segment, sr, mels, hoplen, nfft, filename, start, spectrogram_dir)`
- Creates normalized mel spectrograms and saves as .npy files
- Maintains float32 precision (no conversion to 8-bit)
- Already normalized to [0,1] range
- Saves with .npy extension instead of .png

#### `create_single_spectrogram_npy(segment_info, spectrogram_dir, mels, hoplen, nfft)`
- Creates single spectrograms using the new .npy format
- Parallel to existing `create_single_spectrogram()` function
- Uses `get_spec_npy()` instead of `get_spec_image()`
- Returns same metadata structure for compatibility

### 2. New Notebook Created: `notebooks/AudioSpecting.ipynb`
- Mirror of `AudioProcessing.ipynb` but uses .npy format
- Saves spectrograms to `database/specs/` directory
- Creates `final_specs.csv` metadata file
- Uses `create_spectrograms_from_segments_npy()` function
- Tests SpecAugment with .npy files directly

### 3. Enhanced ModelConfiguring.ipynb:

#### New function: `load_npy_data(specs_dir, specs_csv_path)`
- Loads spectrograms from .npy files
- Reads metadata from `final_specs.csv`
- No /255.0 normalization needed (already normalized)
- Returns features in correct CNN input shape: (N, 1, 224, 313)
- Handles missing files gracefully

#### New data loading cell:
- Uses `specs/` directory and `final_specs.csv`
- Calls `load_npy_data()` instead of `load_csv_data()`

### 4. Updated `utils/util.py`:
- Added exports for `get_spec_npy` and `create_single_spectrogram_npy`
- Maintains backward compatibility

## New Pipeline Flow:

### Old Pipeline:
1. AudioExtracting.ipynb → audio_segments/
2. AudioProcessing.ipynb → spect/ (.png files) + final_spects.csv
3. DataLoading.ipynb → train_data.csv (flattened pixels)
4. ModelConfiguring.ipynb → load_csv_data() (with /255.0 normalization)

### New Pipeline:
1. AudioExtracting.ipynb → audio_segments/
2. **AudioSpecting.ipynb** → **specs/** (.npy files) + **final_specs.csv**
3. **Skip DataLoading.ipynb** (made redundant)
4. ModelConfiguring.ipynb → **load_npy_data()** (no normalization needed)

## Key Improvements:

1. **Preserved Float32 Precision**: No compression to 8-bit PNG
2. **Eliminated Normalization Step**: Data already in [0,1] range
3. **Reduced Pipeline Steps**: Skip CSV conversion of pixels
4. **Maintained Compatibility**: Old functions still exist
5. **Direct NumPy Loading**: Faster than loading images and converting

## Directory Structure:
- `database/specs/` - Contains .npy spectrogram files
- `database/meta/final_specs.csv` - Metadata for .npy files
- `database/spect/` - Still contains .png files (for old pipeline)
- `database/meta/final_spects.csv` - Metadata for .png files (for old pipeline)

## Files Modified:
- `utils/data_processing.py` - Added new functions
- `utils/util.py` - Added exports
- `notebooks/ModelConfiguring.ipynb` - Added new function and loading cell
- `notebooks/AudioSpecting.ipynb` - New notebook created

## Usage:
1. Run `AudioExtracting.ipynb` to create audio segments
2. Run `AudioSpecting.ipynb` to create .npy spectrograms
3. Run `ModelConfiguring.ipynb` using the new data loading cell
4. Training pipeline now uses float32 precision spectrograms

Both pipelines coexist - the old PNG-based pipeline and the new NPY-based pipeline can be used independently.

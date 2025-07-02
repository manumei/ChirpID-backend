**Status Meaning**
- Finished ☑️ | Notebook has been completed and run. Its purpose has been achieved, it does not need to be used or modified anymore
- Developed 🚀 | Notebook logic is basically complete, but notebook hasn't been properly executed, so looking at it still serves purpose
- Work in Progress 🛠️ | Notebook hasn't been completed yet, the logic must continue to be implemented for it to work and be run
- To be started ⏳ | Notebook hasn't yet been started (all notebooks are equally essential, but are usually done in order)

### 1. [MetaDataLoading.ipynb](../notebooks/MetaDataLoading.ipynb) | Status: Finished ☑️

Takes the original train_metadata.csv and makes cutoffs based on coordinates, audio rating, and min_samples per species. Narrows down to a smaller subset of species and samples, and then transcribes them into processed CSVs in database/meta/ for future extraction.

### 2. [AudioLoading.ipynb](../notebooks/AudioLoading.ipynb) | Status: Finished ☑️

Reads the CSV to see which samples from the .ogg audio files to take, and it copies them into the dev/ and test/ folders in database/audio/, doing a stratified sampling to ensure proportional class representation in both datasets.

### 3. [AudioExtracting.ipynb](../notebooks/AudioExtracting.ipynb) | Status: Finished ☑️

Loads the dev/ audio files with librosa, extracts high-energy segments using RMS thresholding with balanced sampling per class, and saves these audio segments as .wav files to database/audio_segments/. Creates a metadata CSV (audio_segments.csv) with information about each extracted segment including class_id, original filename, and segment index. This notebook must be run before AudioSpecting.

### 4. [AudioSpecting.ipynb](../notebooks/AudioSpecting.ipynb) | Status: Finished ☑️

Reads the extracted audio segments from database/audio_segments/ using the metadata CSV created by AudioExtracting notebook. Applies optional noise reduction to each segment, generates mel spectrograms, and saves them as .npy images in database/specs/. Also saves a few test audio samples for verification. Creates final_specs.csv with spectrogram metadata for downstream processing.

### 5. [DataLoading.ipynb](../notebooks/DataLoading.ipynb) | Status: Finished ☑️

Takes the spectrograms from the specs/ folder, and gets the images into vector form with the grayscale of all the pixels, so they can be given to a Fully-Connected Neural Network for training. It loads the pixel info into a CSV so they can then be read & extracted into flat vectors for the FCNN without having to re-run this notebook again. The target CSV (at database/meta/train_data.csv) has a row for each sample, with the columns 'label' (class_id), and then all the pixel elements of the spectrogram.

... add:
- ModelConfiguring.ipynb (for testing parameter configurations)
- ModelBuilding.ipynb (for testing architectures)
- TrainingCNN.ipynb (final training and model saving for the CNN candidates)
- TrainingFCNN.ipynb (final training and model saving for the FCNN candidate)
- ModelTesting.ipynb (final tests with each pre-trained model)
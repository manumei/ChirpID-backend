**Status Meaning**
- Finished ☑️ | Notebook has been completed and run. Its purpose has been achieved, it does not need to be used or modified anymore
- Developed 🚀 | Notebook logic is basically complete, but notebook hasn't been properly executed, so looking at it still serves purpose
- Work in Progress 🛠️ | Notebook hasn't been completed yet, the logic must continue to be implemented for it to work and be run
- To be started ⏳ | Notebook hasn't yet been started (all notebooks are equally essential, but are usually done in order)

### 1. [MetaDataLoading.ipynb](../notebooks/MetaDataLoading.ipynb) | Status: Finished ☑️

Takes the original train_metadata.csv and makes cutoffs based on coordinates, audio rating, and min_samples per species. Narrows down to a smaller subset of species and samples, and then transcribes them into processed CSVs in database/meta/ for future extraction.

### 2. [AudioLoading.ipynb](../notebooks/AudioLoading.ipynb) | Status: Finished ☑️

Reads the CSV to see which samples from the .ogg audio files to take, and it copies them into the dev/ and test/ folders in database/audio/, doing a stratified sampling to ensure proportional class representation in both datasets.

### 3. [AudioExtracting.ipynb](../notebooks/AudioExtracting.ipynb) | Status: Developed 🚀

Loads the dev/ audio files with librosa, extracts high-energy segments using RMS thresholding with balanced sampling per class, and saves these audio segments as .wav files to database/audio_segments/. Creates a metadata CSV (audio_segments.csv) with information about each extracted segment including class_id, original filename, and segment index. This notebook must be run before AudioProcessing.

### 4. [AudioProcessing.ipynb](../notebooks/AudioProcessing.ipynb) | Status: Developed 🚀

Reads the extracted audio segments from database/audio_segments/ using the metadata CSV created by AudioExtracting notebook. Applies optional noise reduction to each segment, generates mel spectrograms, and saves them as .png images in database/spect/. Also saves a few test audio samples for verification. Creates final_spects.csv with spectrogram metadata for downstream processing.

### 5. [DataLoading.ipynb](../notebooks/DataLoading.ipynb) | Status: Developed 🚀

Takes the spectrograms from the spect/ folder, and gets the images into a matrix form, with the grayscale of all the pixels, so they can be given to a Convolutional Neural Network for training. It loads the matrices pixel info into a CSV so they can then be read & extracted, easily reconverted into a feedable matrix without having to re-run this notebook again. The target CSV (at database/meta/train_data.csv) has a row for each sample, with the columns 'label' (class_id), and then all the pixel elements of the spectrogram.

### 5.5 [DevTraining.ipynb](../notebooks/DevTraining.ipynb) | Status: Work in Progress 🛠️

This notebook is an intermediary step before trying the training sweeps, where I test and debug why things are going wrong with the data. I run single fold tests with various small differences to see if there is a small issue causing big consequences that affect the model's development.

### 6. [ModelSweeping.ipynb](../notebooks/ModelSweeping.ipynb) | Status: Work in Progress 🛠️

Makes a Grid Search and Sweeps of different ML techniques and Model Architectures to try and find the best one for final training and production.

### 7. [ModelTraining.ipynb](../notebooks/ModelTraining.ipynb) | Status: Work in Progress 🛠️

This notebook reads the train_data.csv, gets the matrices and turns them into torch tensors, and with that it trains the CNN PyTorch model imported from utils/models.py. It runs the entire training process and then in a final cell saves the model with its weights in a .pth file, so it can be imported and loaded from other files (such as .py or .ipynb files) for instant predictions based on the trained weights, without having to do re-training.

### 8. [ModelTesting.ipynb](../notebooks/ModelTesting.ipynb) | Status: To be started ⏳

Loads the saved model from the ModelTesting Notebook without doing any re-training, and is given the test set to evaluate final performance.

### Useful Auxiliary Files, from [utils/](../utils/)

**[util.py](../utils/util.py)**

Has useful auxiliary functions for the models and loading data, general-purpose utility auxiliary file. Contains functions for audio segment extraction, spectrogram creation, file I/O operations, and training utilities.

**[models.py](../utils/models.py)**

Has the PyTorch CNN Model logic. Important! When the model receives a value to predict, it should of course go through its entire process of being loaded by librosa, then reduce-noise, cut segment and select window for spectrogram, then the grayscale matrix, and then predicting based on that.

### Workflow Notes

The audio processing workflow has been split into two sequential steps:

1. **AudioExtracting**: Extract and save audio segments to disk with metadata
2. **AudioProcessing**: Load saved segments and create spectrograms

This separation allows for better debugging, intermediate result inspection, and the ability to modify spectrogram parameters without re-extracting segments.

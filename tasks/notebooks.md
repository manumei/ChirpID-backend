**Status Meaning**
- Finished ☑️ | Notebook has been completed and run. Its purpose has been achieved, it does not need to be used or modified anymore
- Developed 🚀 | Notebook logic is basically complete, but notebook hasn't been properly executed, so looking at it still serves purpose
- Work in Progress 🛠️ | Notebook hasn't been completed yet, the logic must continue to be implemented for it to work and be run
- To be started ⏳ | Notebook hasn't yet been started (all notebooks are equally essential, but are usually done in order)

### 1. [MetaDataLoading.ipynb](../notebooks/MetaDataLoading.ipynb) | Status: Finished ☑️

Takes the original train_metadata.csv and makes cutoffs based on coordinates, audio rating, and min_samples per species. Narrows down to a smaller subset of species and samples, and then transcribes them into processed CSVs in database/meta/ for future extraction.

### 2. [AudioLoading.ipynb](../notebooks/AudioLoading.ipynb) | Status: Finished ☑️

Reads the CSV to see which samples from the .ogg audio files to take, and copies them into database/assets/raw/ for future processing

### 3. [WavLoading.ipynb](../notebooks/WavLoading.ipynb) | Status: Finished ☑️

Following train_data.csv, it takes the given .ogg files in raw/ and converts them to 32kHz waveform (.wav) files, splitting them into folders for each species' class_id for later stratifying when doing train-test split.

### 4. [WavProcessing.ipynb](../notebooks/WavProcessing.ipynb) | Status: Work in Progress 🛠️

Takes the waveform audio files, and creates reduced-noise spectrograms on a set window (currently librosa default, soon to be changed), and then loads the spectrograms into database/assets/spect/

### 5. [DataLoading.ipynb](../notebooks/DataLoading.ipynb) | Status: To be started ⏳

Takes the spectrograms from the spect/ folder, and gets the images into a matrix form, with the grayscale of all the pixels, so they can be given to a Convolutional Neural Network for training. It loads the matrices pixel info into a CSV so they can then be read & extracted, easily reconverted into a feedable matrix without having to re-run this notebook again. The target CSV (at database/meta/final_model_data.csv) has a row for each sample, with the columns 'label' (class_id), and then all the pixel elements of the spectrogram.

### 6. [ModelTraining.ipynb](../notebooks/ModelTraining.ipynb) | Status: To be started ⏳

This notebook reads the final_model_data.csv and trains the CNN PyTorch model. It runs the entire training process with cross-validation. And then in a final cell, saves the model with its weights, so it can be imported and loaded from other files (such as .py or .ipynb files) for instant predictions based on the trained weights, without having to do re-training.

### 7. [ModelTesting.ipynb](../notebooks/ModelTesting.ipynb) | Status: To be started ⏳

Loads the saved model from the ModelTesting Notebook without doing any re-training, and is given the test set to evaluate final performance. 

### Useful Auxiliary Files, from [utils/](../utils/)

**util.py**

Has useful auxiliary functions for the models and loading data, general-purpose utility auxiliary file.

**models.py**

Has the PyTorch CNN Model logic. Important! When the model receives a value to predict, it should of course go through its entire process of converting to .wav, then reduce-noise and select window for spectrogram, then the grayscale matrix, and then predicting based on that.

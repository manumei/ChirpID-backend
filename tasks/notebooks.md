**Status Meaning**
- Finished ☑️ | Notebook has been completed and run. Its purpose has been achieved, it does not need to be used or modified anymore
- Developed 🚀 | Notebook logic is basically complete, but notebook hasn't been properly executed, so looking at it still serves purpose
- Work in Progress 🛠️ | Notebook hasn't been completed yet, the logic must continue to be implemented for it to work and be run
- To be started ⏳ | Notebook hasn't yet been started (all notebooks are equally essential, but are usually done in order)

### 1. [MetaDataLoading.ipynb](../notebooks/MetaDataLoading.ipynb) | Status: Finished ☑️

Takes the original train_metadata.csv and makes cutoffs based on coordinates, audio rating, and min_samples per species. Narrows down to a smaller subset of species and samples, and then transcribes them into processed CSVs in database/meta/ for future extraction.

### 2. [AudioLoading.ipynb](../notebooks/AudioLoading.ipynb) | Status: Finished ☑️

Reads the CSV to see which samples from the .ogg audio files to take, and it copies them into the dev/ and test/ folders in database/audio/, doing a stratified sampling to ensure proportional class representation in both datasets.

### 3. [AudioProcessing.ipynb](../notebooks/AudioProcessing.ipynb) | Status: Finished ☑️

Loads the dev/ audio files with librosa, keeping only the segments of high energy audio, creating samples from the division of longer audios into multiple of such segments. It then creates spectrograms on a set window for each segments, and loads them as .png images into database/spect/

### 4. [DataLoading.ipynb](../notebooks/DataLoading.ipynb) | Status: Work in Progress 🛠️

Takes the spectrograms from the spect/ folder, and gets the images into a matrix form, with the grayscale of all the pixels, so they can be given to a Convolutional Neural Network for training. It loads the matrices pixel info into a CSV so they can then be read & extracted, easily reconverted into a feedable matrix without having to re-run this notebook again. The target CSV (at database/meta/final_model_data.csv) has a row for each sample, with the columns 'label' (class_id), and then all the pixel elements of the spectrogram.

### 5. [ModelTraining.ipynb](../notebooks/ModelTraining.ipynb) | Status: To be started ⏳

This notebook reads the final_model_data.csv and trains the CNN PyTorch model. It runs the entire training process with cross-validation. And then in a final cell, saves the model with its weights, so it can be imported and loaded from other files (such as .py or .ipynb files) for instant predictions based on the trained weights, without having to do re-training.

### 6. [ModelTesting.ipynb](../notebooks/ModelTesting.ipynb) | Status: To be started ⏳

Loads the saved model from the ModelTesting Notebook without doing any re-training, and is given the test set to evaluate final performance. 

### Useful Auxiliary Files, from [utils/](../utils/)

**[util.py](../utils/util.py)**

Has useful auxiliary functions for the models and loading data, general-purpose utility auxiliary file.

**[models.py](../utils/models.py)**

Has the PyTorch CNN Model logic. Important! When the model receives a value to predict, it should of course go through its entire process of being loaded by librosa, then reduce-noise, cut segment and select window for spectrogram, then the grayscale matrix, and then predicting based on that.

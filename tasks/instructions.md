# Instructions

## 1. Data Processing

### 1.0 MetaData Loading

1.0.0) Open the MetaDataLoading.ipynb Jupyter Notebook

1.0.1) Select the (Latitude, Longitude) and Rating cutoffs for the data to be loaded

1.0.2) Run All Cells in the Jupyter Notebook, and voila. Loading is done!

### 1.1 Audio Loading

Run All Cells from the AudioLoading.ipynb, after completing step 1.0, all .ogg audios have now been loaded into their respective folders in database/audio/dev & database/audio/test, done with the stratified sampling and all, into an 80-20 split.

### 1.2 Audio Processing

Run All Cells from AudioProcessing.ipynb, it will load the files at 32kHz frequency, segment them into high-energy segments of a fixed duration with RMS threshold, then get the spectrograms with a set window from all the samples. The spectrograms will be loaded into database/spect/dev/ and ready for data extraction and loading.

### 1.3 Data Loading

Run All Cells from DataLoading.ipynb, it will load a CSV with the matrices of the grayscale pixels from the dev spectrograms, attached with the class_id label belonging to the given bird sample. The CSV will be loaded at dataset/samples/train_data.csv

## 2. Running the Model

### 2.1 Model Training

Run All Cells from ModelTraining.ipynb, this fetches the CSV with the 

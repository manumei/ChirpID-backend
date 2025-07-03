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

### 6. [ModelConfiguring.ipynb](../notebooks/ModelConfiguring.ipynb) | Status: Finished ☑️

Tests different parameter configurations for neural network models. Experiments with various hyperparameters like learning rates, batch sizes, optimizers, and regularization techniques to find optimal configurations before full model training. This notebook helps identify the best parameter settings for each model architecture.

### 7. [ModelBuilding.ipynb](../notebooks/ModelBuilding.ipynb) | Status: Finished ☑️

Focuses on testing different neural network architectures for bird classification. Experiments with various model designs including different layer configurations, activation functions, and architectural patterns like residual connections, attention mechanisms, and ensemble approaches to determine the most effective model structures.

### 8. [ModelTweaking.ipynb](../notebooks/ModelTweaking.ipynb) | Status: Finished ☑️

Performs final optimization and fine-tuning on the top-performing model candidates identified from previous experiments. Applies advanced techniques like learning rate scheduling, data augmentation, and model ensembling to squeeze out the best possible performance from the most promising architectures.

### 9. [SweepingFCNN.ipynb](../notebooks/SweepingFCNN.ipynb) | Status: Finished ☑️

Conducts comprehensive hyperparameter sweeps and systematic evaluation of Fully-Connected Neural Network (FCNN) variants. Tests multiple FCNN architectures (v0 through v10) with different configurations to identify the best-performing FCNN model. Includes performance comparison, model saving, and detailed analysis of results across different network designs.

### 10. [TrainingCNN.ipynb](../notebooks/TrainingCNN.ipynb) | Status: Finished ☑️

Handles the final training and model saving for the best Convolutional Neural Network (CNN) candidates. Trains the selected CNN architectures on the full dataset with optimized parameters, implements proper validation procedures, and saves the trained models for deployment and testing.

### 11. [TrainingFCNN.ipynb](../notebooks/TrainingFCNN.ipynb) | Status: Finished ☑️

Manages the final training and model saving for the best-performing Fully-Connected Neural Network (FCNN) candidate identified from the sweeping process. Trains the selected FCNN architecture with optimal hyperparameters, validates performance, and saves the final trained model for production use.

### 12. [ModelTesting.ipynb](../notebooks/ModelTesting.ipynb) | Status: Finished ☑️

Conducts comprehensive evaluation and testing of all pre-trained models (both CNN and FCNN variants). Performs final validation on test datasets, generates performance metrics, confusion matrices, and comparison analyses. This notebook provides the final assessment of model performance and helps select the best model for deployment.

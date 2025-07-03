# Info

## Plans

### Data Processing

##### Data Selection

Make the cutoffs with Regions & Ratings. Load the selected .ogg files into database/raw/
Reload the metadata CSV to keep only the selected samples

##### Metadata Loading

Map the bird_name to integers, class IDs
Create list of (filename, clas_id) entries

##### Train Test Split

Split data into Train & Test, *with stratify* so all classes get in both, using SK-Learn modules. Final data should be lists of tuples [(filename1, class_id1), (filename2, class_id2), ...].

##### Load the Audios

Using torchaudio.load or librosa.load, setting frequencies to standard 32kHz (librosa o torchaduil.transforms ambos lo hacen)

##### Silence Filtering

Filter out the low-energy background noise and silence, to only keep the relevant bird chirp parts, using segments and rms thresholds.
Decide fixed duration window in seconds. Choose windows that contain enough audio energy/amplitude (to ensure bird sound).

##### Spectrograms

Apply Fourier to get the Spectrograms, amplitude to decibels, etc. Creemos una carpeta con las imagenes de los spectrograms asi no tenemos que seguir lidiando con los audios. Then get the matrix for the grayscale of the spectrogram for the model. testingLibrosa.py

##### Normalize Data

Fijarse despues bien como hacerlo. Pregunta perfecta para hacerle a Santi Pierini honestly, or just ask Chat and check in the papers to see how they do it, and on what data (maybe standardize the final vectors)

##### Pass the Data

Convert to PyTorch tensors, check the shapes and such. Then create DataLoader, do shuffle=True and consider mini-batching via batch_size and stuff, and get on to Training Time

#### Final DataProcessing()

Once every function is finished, reset the changes (the .wav folder and such variables)

---

## Problems & Solutions

### Data & Training

##### Splits might miss some classes

Use stratify, ask ChatGPT, use sklearn split with stratify. Stratify ensures classes areproportionally represented

##### Amount of samples across species is unbalanced

Set a fixed amount of samples per species target.
For species exceeding that target, undersample (cut samples).
For species below that target, instead of SMOTE or duplication techniques, just take chrip samples from long audios if possible.

##### Cross Validation es medio quilombo

SK-Learn Modules for K-Fold, turn into torch.tensors for the model

##### Testing & Inference Samples have varying durations and might be full of noise/silence segments

When doing inference and prediction, the model just receives an unprocessed .ogg file. The model then processes this .ogg file, it loads it at 32kHz, divides into all the high-energy segments (of the fixed duration), then for each segment, gets the spectrogram of the given fourier window, and turns each spectrogram into data matrix format, and gives each matrix to the model for inference. If there is more than 1 segment to predict from, then the model receives all the matrices and calculates the probabilities for each. With these probabilities, doing a mean softmax aggregation of each $p_i$ to get the final probabilities, combining all the segments as part of the inference. So the longer the audio, the more it segments it gets to predict. 

---

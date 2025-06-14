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

Splits might miss some classes

Use stratify, ask ChatGPT, use sklearn split with stratify. Stratify ensures classes areproportionally represented

Amount of samples across species is unbalanced

Set a fixed amount of samples per species target.
For species exceeding that target, undersample (cut samples).
For species below that target, instead of SMOTE or duplication techniques, just take chrip samples from long audios if possible.

Cross Validation es medio quilombo

SK-Learn Modules for K-Fold, turn into torch.tensors for the model

Windows might be taken of full noise/silence

Choose a fixed time interval (eg. SILENCE_CUT= 2.5 seconds), and then Cut & RePaste the .wav files, but having removed from each audio, every section where there are 2.5s+ of noise / silence.

Analyze with the spectrograms, check with ChatGPT how to do it optimally. Reload the audios with the silence/noise parts cut. So now data augmentation can be done comfortably.

Make sure that the window chosen later is longer than the SILENCE_CUT chosen here. For example, try SILENCE_CUT = 3, AUDIO_WINDOW = 5 so that no windows are mostly noise/silence.


---

# Info

## Plans

### Data Processing

**Data Selection**<br>
Make the cutoffs with Regions & Ratings. Load the selected .ogg files into database/raw/
Reload the metadata CSV to keep only the selected samples

**Metadata Loading**<br>
Map the bird_name to integers, class IDs
Create list of (filename, clas_id) entries

**Train Test Split**<br>
Split data into Train & Test, *with stratify* so all classes get in both, using SK-Learn modules.<br>
Final data should be lists of tuples [(filename1, class_id1), (filename2, class_id2), ...]

**OGG to WAV**<br>
For better work with transforms, place the wavs in database/assets/wav

**Load the Audios**<br>
Using torchaudio.load or librosa.load, setting frequencies to standard 32kHz (librosa o torchaduil.transforms ambos lo hacen)

**Silence & Noise Removal**<br>
Choose a fixed time interval (eg. SILENCE_CUT= 2.5 seconds), and then Cut & RePaste the .wav files, but having removed from each audio, every section where there are 2.5s+ of noise / silence.<br>
Analyze with the spectrograms, check with ChatGPT how to do it optimally. Reload the audios with the silence/noise parts cut. So now data augmentation can be done comfortably.<br>
Make sure that the window chosen later is longer than the SILENCE_CUT chosen here. For example, try SILENCE_CUT = 3, AUDIO_WINDOW = 5 so that no windows are mostly noise/silence.<br>

**Silence Filtering**<br>
Filter out the low-energy background noise and silence, to only keep the relevant bird songs.<br>
However, fijarse de que pueda servir cuando se le de una grabacion con cierto silencio o ruido entrelazado. Audio energy (testingLibrosa.py).<br>
Ask ChatGPT &/ Santi Pierini, and check the papers how that second part would work.

**Duration Handling**<br>
Decide fixed duration window (hay que ver cuantos segundos)
Cut the longer audios, chose windows that contain enough audio energy/amplitude (to ensure bird sound)

**Spectrograms**<br>
Apply Fourier to get the Spectrograms, amplitude to decibels, etc.<br>
Creemos una carpeta con las imagenes de los spectrograms asi no tenemos que seguir lidiando con los audios.<br>
Then gt the matrix for the grayscale of the spectrogram for the model. testingLibrosa.py

**Normalize Data**<br>
Fijarse despues bien como hacerlo. Pregunta perfecta para hacerle a Santi Pierini honestly, or just ask Chat and check in the papers to see how they do it, and on what data (maybe standardize the final vectors)

**Pass the Data**<br>
Convert to PyTorch tensors, check the shapes and such. Then create DataLoader, do shuffle=True and consider mini-batching via batch_size and stuff, and get on to Training Time

**Final DataProcessing()**<br>
Once every function is finished, reset the changes (the .wav folder and such variables)

---

## Problems & Solutions

### Data & Training

**Splits might miss some classes**
Use stratify, ask ChatGPT, use sklearn split with stratify. Stratify ensures classes areproportionally represented

**Amount of samples across species is unbalanced**
Set a fixed amount of samples per species target.
For species exceeding that target, undersample (cut samples).
For species below that target, instead of SMOTE or duplication techniques, just take chrip samples from long audios if possible.

**Cross Validation es medio quilombo**
SK-Learn Modules for K-Fold, turn into torch.tensors for the model

---

## Ideas

Suboscines y Oscines


# Pipeline

My current processing pipeline for training is:

**Data Fractioning**
1. Get a fraction of the dataset, keeping only birds in Argentina
2. Keep only samples where rating > 3
3. Keep only species where I have 10+ samples
4. Load all the audios that meet these conditions
5. Split them into dev/test

**Audio Processing (only for Dev set)**
6. Load the audios with librosa
7. Cut into segments of 5 seconds, only keep the segments with high enough RMS energy, add them as new samples
8. From all the 5-second samples, create grayscale mel-spectrograms

**Data Loading (only for Dev set)**
9. Load the spectrograms and get their grayscale pixels in matrix shape
10. Split into Train/Val
11. Load as PyTorch tensors
12. Feed to the PyTorch CNN for training

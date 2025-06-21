# Pipeline

My current processing pipeline for training is:

**Data Fractioning**
1. Get a fraction of the dataset, keeping only birds in a certain region around Argentina
2. Keep only samples with high rating
3. Keep only species where I have several samples and segments
4. Keep only species where I have enough unique authors
5. Load all the audios that meet these conditions
6. Split them into dev/test, grouping by author

**Audio Processing (only for Dev set)**
7. Load the audios with librosa
8. Cut into segments of 5 seconds, only keep the segments with high enough RMS energy, add them as new samples
9. From all the 5-second samples, create grayscale mel-spectrograms

**Data Loading (only for Dev set)**
10. Load the spectrograms and get their grayscale pixels in matrix shape
11. Split into Train/Val grouping by author
12. Load as PyTorch tensors
13. Feed to the PyTorch CNN for training

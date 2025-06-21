I want to modify the load_spectrograms function to limit the max amount of final samples per each class_id to a cap such as 100. However, if I just do segments of each audio until I reach the cap, then I will likely reach the 100 samples without going through each audio, and a few audios will hoard the dataset with all their segments.

Would it be a feasible alternative to instead do something such as:
- Establish a list of all audios to iterate through
- First go over every audio, analyze the segment from 0 to 5s
- Then go over every audio again, analyze from 5s to 10s
- Every audio again, analyze from 10s to 15s
- etc

2 cases when this stops:
- When an audio's duration falls short of the segment being checked, the audio is removed from the list of audios to keep iterating.
- When an audio of a certain species (class_id) gets its turn, but already the cap=100 segments has been exceded by its species. Then remove it from the list too.

And when the list is empty, the loading is done.

I was thinking a better way to do this would be also splitting the audio segmenting part and the spectrogram_loading part. Make 2 functions, one that takes the audios, calculates the RMS and threshold and keeps all the final segments, writing them into a new folder (call it load_segments). Then another function that reads the directory with all the segments, and creates and loads the grayscale log-mel spectrograms for each (call it load_spectrogram). Since our current code does both things at the same time
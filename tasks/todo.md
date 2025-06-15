# To-Do

How to add noise to make model more robust? Read papers and ask Chat/Claude (also make them read papers).
How to make the model deal with noise/silence? Should the model be able to predict if the audio is *not a bird*?
What if the model receives just silence? What if it receives some audio that just isn't a bird at all? Shazam says "song not found".

Try re-adding noisereduce for the spectrograms (and test audios!), band-pass filter to remove background noise. Might make the model better.
Try doing it in spect2/ and call that folder for the noise-reduced spectrograms instead of spect/, so make train_data and train_data2.

# Noise Management

Should I add noise to make model more robust? What are the common practices in tasks like this one? How did the papers do it?. I already use noisereduce when I load the audios for the spectrograms, is this enough to deal with noisy inputs for inference?

Also another question which might or might not be related: How to make the model deal with noise/silence/fake samples? Should the model be able to predict if the audio is *not a bird*? What if the model receives just silence? What if it receives some audio that just isn't a bird at all? Shazam for example says "song not found"? What are the common practices? Do any of the papers or BirdNet mention how they handle this?

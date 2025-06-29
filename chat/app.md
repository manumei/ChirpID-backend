# For audio_processing.py

The audio_process() function being imported from data_processing.py as you can see, takes a single audio path and returns a list of all grayscale log-mel-spectrogram pixel matrices for each usable segment in the audio, so you can convert it into a tensor for my CNN and perform inference. I have the model weights saved in a .pth file at ../models/bird_cnn.pth

I want you to make a function that receives the path to the audio and the path to the model weights, it calls audio_process() to extract the list of matrices out of the audio, and then loads up the PyTorch CNN model weights, and iterates the list of matrices, making an inference on each of the segments (each matrix), with the softmax output, and returns the average softmax probabilities for each class (average between each segment).

If you see it as more optimized or better-structured, you can make 1 or more extra auxiliary sub-functions. I want you to only edit the code in audio_processing.py, use the other files as references of how we do this and how each function being called actually works, but there is no need to modify those right now.

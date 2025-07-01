# Test

Please edit ModelTesting.ipynb to set up proper inference of my test set. I want to use the perform_audio_inference() function from inference.py, which let me take an audio file path, and the respective model class and weights to be used, and performs the inference on the audio file to return the softmax probabilities of each class.

I want ModelTesting.ipynb to iterate through the audio paths in the ../database/audio/test/ directory, and for each of them, perform the inference, compare with the true label, and then print the total evaluation metrics (Cross Entropy Loss, F1 Score, Accuracy, Confusion Matrix) using the functions from metrics.py.

To compare with the true label for each test file, refer to ../database/meta/test_data.csv, which contains for each sample, a 'filename' and 'class_id' column, where filenames might be for example XC1540.ogg for the audio at ../database/audio/test/XC1540.ogg, and has its respective numeric label in class_id of course.

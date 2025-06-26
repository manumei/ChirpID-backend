# Specs to .npy

Convert my spectrogram pipeline from .png to .npy format
I currently generate mel spectrograms and save them as 8-bit PNG images, then load them back for CNN training. I want to switch to saving as .npy files to preserve float32 precision and avoid the /255.0 normalization step.

Currently the spectrogram pipeline is:
- (In AudioProcessing.ipynb) | Calculate the log-mel spectrograms
- (In AudioProcessing.ipynb) | Generate the spectrograms imgs as .pngs by calling data_processing.py in AudioProcessing.ipynb
- (In AudioProcessing.ipynb) | Write a final_spects.csv with the .png filenames, the author, class_id
- (In DataLoading.ipynb) | Load the images and write the 8-bit .png pixels into a CSV
- (In ModelConfiguring.ipynb) | Read the CSV in ModelConfiguring.ipynb, normalize to [0,1] and reshape from the flat vectors

The new Spectrogram pipeline I want is:
- Calculate the log-mel spectrograms **exactly like im currently doing**
- (In AudioSpecting.ipynb) | Normalize to [0,1] and save the images as float32 .npy instead to avoid compression, save them in specs/ instead of spect/
- (In AudioSpecting.ipynb) | Write a final_specs.csv with the .png filenames, the author, class_id
- (In AudioSpecting.ipynb) | SKIP THE CSV STEP.
- (In ModelConfiguring.ipynb) | Load the .npy imgs from the specs/ folder directly in ModelConfiguring.ipynb, reshape for the CNN

Key new functions:
get_spec_npy(), the new equivalent of get_spec_image()
load_npy_data(), a new function implemented directly inside a cell in ModelConfiguring.ipynb, called in the later cell to get the data, acts as the new equivalent of load_spect_data, but skipping the /255.0 step and working with the specs/ directory and instead of the CSV

Generate mel spectrogram → normalize to [0,1] → save as float32 .npy
Load .npy → reshape for CNN (no normalization needed)

Please add these functions to use .npy format while maintaining the same functionality and data flow.
With the important note that we do not modify the functions of our current pipeline, but rather, add new ones.
So we add 

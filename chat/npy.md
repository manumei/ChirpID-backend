# Specs to .npy

Convert my spectrogram pipeline from .png to .npy format
I currently generate mel spectrograms and save them as 8-bit PNG images, then load them back for CNN training. I want to switch to saving as .npy files to preserve float32 precision and avoid the /255.0 normalization step.

Currently the spectrogram pipeline is:
- (In AudioProcessing.ipynb) | Calculate the log-mel spectrograms
- (In AudioProcessing.ipynb) | Generate the spectrograms imgs as .pngs by calling data_processing.py in AudioProcessing.ipynb
- (In AudioProcessing.ipynb) | Write a final_spects.csv with the .png filenames, the author, class_id
- (In DataLoading.ipynb) | Load the images and write the 8-bit .png pixels into a CSV
- (In ModelConfiguring.ipynb) | Read the CSV in ModelConfiguring.ipynb, normalize to [0,1] and reshape from the flat vectors. This is done with the load_csv_data() function.

The new Spectrogram pipeline I want is:
- Calculate the log-mel spectrograms **exactly like im currently doing**
- (In AudioSpecting.ipynb) | Normalize to [0,1] and save the images as float32 .npy instead to avoid compression, save them in specs/ instead of spect/
- (In AudioSpecting.ipynb) | Write a final_specs.csv with the .png filenames, the author, class_id
- (In AudioSpecting.ipynb) | SKIP THE CSV STEP.
- (In ModelConfiguring.ipynb) | Load the .npy imgs from the specs/ folder directly in ModelConfiguring.ipynb, reshape for the CNN. Done with the new load_npy_data() function.

Key new functions:
- (in audio_processing.py) | get_spec_npy(), the new equivalent of get_spec_image()
- (in ModelConfiguring.ipynb) | load_npy_data(), a new function implemented directly inside a cell in ModelConfiguring.ipynb, called in the later cell to get the data, acts as the new equivalent of load_csv_data(), but skipping the /255.0 step and working with the specs/ directory and the final_specs.csv instead of the train_data.csv (which currently holds pixels and metadata).csv,

Please add these functions to use .npy format while maintaining the same functionality and data flow. With the important note that we do not modify the functions of our current pipeline, but rather, add new ones.

So we add the new get_spec_npy to audio_processing _without removing_ get_spec_image(). And then we create AudioSpecting.ipynb, to reflect/imitate exactly what AudioProcessing.ipynb does, but calling get_spec_npy instead of get_spec_image() (yes, most of it will be duplicate code between AudioProcessing.ipynb and AudioSpecting.ipynb, yes I am fine with it)

Finally, in ModelConfiguring.ipynb, we replace calling load_csv_data() that currently deals only with the CSV, with load_npy_data(), which will call on the specs directory and final_specs.csv to get the features, labels and authors directly from there. We define load_npy_data IN A NEW CELL without deleting the cell that defines load_csv_data (just making it redundant for now, I may use it later).

Do not modify DataLoading.ipynb! We only make it redundant via not calling its outputs (for now, as I will use it in other future tests). If you find any imperfections or optimizations in parts of the hierarchy/pipeline that are affected by this change, then please fix them or improve them.

Once all the changes have been done, please list me a brief summary reporting all the changes and how this works for now.

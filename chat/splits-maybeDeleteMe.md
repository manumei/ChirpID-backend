### Splits Prompt??

#### Separate splits from training? Unnecessary? Pinchovich? Delete me?

As you know, we have mostly a completely top-down modular structure for our project, especially for training, where the user goes to the training notebooks like ModelConfiguring, and just calls on either of the core training functions from training_core.py to work down the established modular hierarchy from a single top call.

However, there is one thing I would like to keep at separate calls, which is the data loading for the training and the training itself. I want to first call the best split function on a cell, save the variables, and on a different cell below, call the training functions from training_core.py

# Splitting Before Training

We have almost completed our beautiful top-down modular set up, where all the utility functions in the various auxiliary python files in utils/ are connected to the hierarchy, so the user can call on either of the training_core.py functions (single-fold or k-fold training) with the desired configuration, and run the training and then print its metrics, like exemplified most importantly in ModelConfiguring.ipynb.

However, there is one key improvement we have to make. As we sweep for configurations, we run the training function with each different one, and call the training function. This has one key problem, which is that the training functions call inside them the split searching functions for the author-grouped splits such as search_best_group_seed_kfold. This is a function that should be completely independent of the techniques of the model being used, and dependent only on the data, yet, we are loading it again and again on every single configuration, which takes up an enormous amount of time considering how heavy the function gets with high max_attempts.

So, while I do love keeping every auxiliary function within the top-down hierarchy of the two training functions in training_core, perhaps we should make a single exception for the best group seed functions from split.py. We should take them out of the core training functions, call them in a previous cell of the notebook (ModelConfiguring.ipynb), and then adjust the core training functions to work by receiving the splits, whether it is k-fold splits or single splits.

If you consider this an improvement, please update all the necessary code. Especial focus of course on editing training_core.py and ModelConfiguring.ipynb, and revising (maybe detailing) split.py. But of course, any other edits that improve this are welcome.

When you are finished, list a brief summary of the changes made.

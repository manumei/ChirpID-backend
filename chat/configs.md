Edit the almost-empty ModelConfiguring.ipynb. Set it up thoroughly so I can use it to define specific "configs" where I can try different combinations of parameters for models. I want to be able to sey up 'configs', meaning dictionaries of all the parameters I can vary to test the best model possible. I want to be able to test and modify the following parameters in each configuration:

- ADAM Optimizer: bool
- estop_thresh: int
- batch_size: int
- use_class_weights: bool
- l2: float
- lr_sched: {'exp', 'redPlateau', 'cosAnneal'} (each with their respective hyperparameters if true)
- initial learning rate: float
- standardize: bool
- specAugment: bool
- noiseAugment: bool

Edit or add all the necessary auxiliary functions from the utility files in utils/ as you might see fit. The ModelConfiguring notebook should start like the other Training Notebooks, with the Imports cell, then the Data Processing and Splitting cells.

In this case you then add the Configurations (configs) Cell where N set-ups named config0, config1, config2, etc. are defined with varying takes on the hyperparameters (for now lets start off with 20 template configs). Then comes the Training cell where a 'for' loop iterates through the configurations, calling the model with each respective combination of hyperparameters, storing the results. And lastly, plotting and displaying each of the achieved results by each configuration (with config ID) so the user can determine which worked best.

For the initial 20 configs to try as a template, try configurations that fit for a CNN that predicts audio species based on sound, using around 3200-800 train/val samples in its process, with each sample being the matrix of a 313x224 grayscale log-mel spectrogram, and around 30 classes to predict.

Remember to mantain and make use of the very clear top-down modular structure we have set up, especially for the Training process, passing on configurations cleanly to the core training functions, as the top step of the hierarchy we have set up, so that pipeline should mantain the same principles it currently has.

Once done, write a report listing the changes made, and write a brief markdown file at tasks/configs.md, where you:
- Describe what each parameter does
- Explain how to choose each parameter properly
- Briefly show which parameters heavily rely on others, so the user doesn't take wrong conclusions from the configuring (for example, cannot modify lr_sched without considering learning rate)
- Briefly explain what to look for and to expect of each hyperparameter for the current project (3200 samples, 30 classes, 313x224 spectrograms)
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

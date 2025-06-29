# Config Parameters

Please make sure the training uses the given configurations. For exampke, on line 348 of training_engine.py, it uses the ReduceLROnPlateau, but I in fact pass on the type of lr scheduler I want in the configs, and its never being used because training_engine doesnt call for it, it just uses ReduceLROnPlateau always, and I dont want that. I want to use what my config says for whatever argument I have defined in my configs (only use a default if it is NOT defined in my configs in ModelConfiguring.ipynb)

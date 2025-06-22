# Noise Management

Implement on-the-fly spectrogram augmentation for my model training pipeline, using a combination of SpecAugment (time masking, frequency masking, and optionally time warping) and additive Gaussian noise. What you need to implement is the following:

- Apply augmentations only during training batches, not during initial spectrogram creation or validation/testing.
- SpecAugment should include at least frequency masking and time masking. (Time warping is optional depending on complexity or library support.)
- Additive Gaussian noise should be applied with configurable standard deviation, after SpecAugment, to the batch or sample.
- Implementation should work within my existing PyTorch DataLoader/Dataset structure, applying transformations inside __getitem__ or via a callable transform.
- Ensure that augmentation is randomized for each epoch/batch.
- Use only standard libraries (torch, numpy, etc.) or minimal external dependencies if required.
- Integrate with my current codebase and data pipeline as described in the provided files.
- Clearly document where in the pipeline the augmentations are applied.
- Output only the implementation code and brief integration notes. No commentary or extra explanation.

*IMPORTANT! I want both of these as optional booleans before training the model itself. The training function must receive the parameters ```specAugment: bool```, ```gaussianNoise: bool```, and use the True/False conditions to determine whether or not to apply any of the two augmentation methods.*

And as always, implement this in a way that follows the top-down modular structure of our training process. The pipeline should stay like the one we have constructed and detailed in tasks/train_instruct.md. Reminder, we already have a specaugment.py file, so most of the work for this implementation will happen in that file. Then make sure that the file is properly called and used by the rest of the hierarchy above it, with the top of the pyramid for the user being the boolean conditions of ```specAugment: bool``` and ```gaussianNoise: bool``` in the final training function taken from training_core.py. Make sure you implement the logic and then propagate it up so it can be properly used.

When finished, list a brief and concise summary of how this implementation works on each scale of the hierarchy.

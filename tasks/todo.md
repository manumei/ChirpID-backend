# To-Do

How to add noise to make model more robust? Research what the common pracvtices are, research how the papers did it.

How to make the model deal with noise/silence? Should the model be able to predict if the audio is *not a bird*? What if the model receives just silence? What if it receives some audio that just isn't a bird at all? Shazam says "song not found".

Balance the classes. Use cost-reweighting, applied directly to the PyTorch tensors and to the criterion (cross-entropy). Other methods of balancing are not good for audios and spectrograms. The papers that mention dealing with class imbalance, all use cost re-weighting.

Try limiting the max amount of segments per single audio sample, so that a single very long recording doesn't get significant over-representation above the other birds in that same (como dijo Trini, que la pajarita "Maria" no tenga demasiada sobre-representacion en toda su especie).

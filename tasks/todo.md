# To-Do

How to add noise to make model more robust? Research what the common pracvtices are, research how the papers did it.

How to make the model deal with noise/silence? Should the model be able to predict if the audio is *not a bird*? What if the model receives just silence? What if it receives some audio that just isn't a bird at all? Shazam says "song not found".

Balance the classes. Use cost-reweighting, applied directly to the PyTorch tensors and to the criterion (cross-entropy). Other methods of balancing are not good for audios and spectrograms. The papers that mention dealing with class imbalance, all use cost re-weighting.

Try limiting the max amount of segments per single audio sample, so that a single very long recording doesn't get significant over-representation above the other birds in that same (como dijo Trini, que la pajarita "Maria" no tenga demasiada sobre-representacion en toda su especie).

(De mas esta aclarar), try sweeps and grid searches of different architectures, optimization and regularization techniques. Try L2 Reg, try using vs not using ADAM, try learning rate momentum (todas las 'vistas en la cursada').

Implement Early Stopping (xd), de mas aclarar again, para no correr 300 epochs de mas, tirar tiempo, y overfittear while at it. Research if there's some common heuristic for what early-stop margin to set. Replicar lo que pensé para el TP3 de guardarse el valor de loss mas bajo y sus pesos en esa instancia, y si *ese* valor no se mejora despues de $n$ epocas, entonces ahi si se corta el training.

User cares about accuracy, not cross-entropy loss. Cuando me ponga a correr los barridos y grid search, uso la cross-entropy loss para entrenar y penalizar al modelo, pero para elegir el modelo final para deploy, evaluar el que tenga mejor accuracy.

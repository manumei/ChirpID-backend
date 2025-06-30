# To-Do

**Fully-Connected Neural Network** (Lara)
Implement another model in models.py, an FCNN that works with flattened vectores instead of matrices.
Aca dejo lo quepuse en el mensaje de WhatsApp por si sirve de tip:

- En ModelTraining.ipynb | Agrega en la celda donde se define features, antes del features.reshape(313, 224), agrega una variable features_fcnn que sea las features pero flattened, y despues fijate de repetir oo mismo que hice con features de los torch.tensors y eso, pero con features_fcnn

- En util.py | Agrega en la funcion audio_processing, un parametro boolean, fcnn=false por default. Y cerca del final de la funcion, donde hace matrices.append(matrix) o algo asi al final del loop for agrega un

    ```python
    if fcnn:
       vector = matrices.flatten()
       matrices.append(vector)
Asi retornea una lista de vectores flats, en vez de matrices, como necesita la fcnn

Despues en models.py crea el modelo, uno nuevo abajo del que ya tenemos de BirdCNN. Avisale a Chat que considere que las dimensiones de los vectores de features de input son (70,112)

**Barridos & Grid Search** (Mei)
Once model is defined:
Instead of just varying techniques and architectures, also vary some arbitrary values for data-processing.
Try varying the threshold_factor, the segment durations, and maybe even some others? hop_len, nfft, mel_bins? might be too much though.

User cares about final predictions, not cross-entropy loss. Cuando me ponga a correr los barridos y grid search, uso la cross-entropy loss para entrenar y penalizar al modelo, pero para elegir el modelo final para deploy, evaluar el que tenga mejor F1 Score.

**Save Models to models/ and their mappings to mapping/**
Save the model weights as pth files, and, (important!) save the respective mappings too. Since class_mapping.csv changes dynamically. We must preserve the current state of the mappings for each saved model.

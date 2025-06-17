# To-Do

How to add noise to make model more robust? Research what the common pracvtices are, research how the papers did it.

How to make the model deal with noise/silence? Should the model be able to predict if the audio is *not a bird*? What if the model receives just silence? What if it receives some audio that just isn't a bird at all? Shazam says "song not found".

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

~~**Limit Segments per Audio** (Mei)
Try limiting the max amount of segments per single audio sample, so that a single very long recording doesn't get significant over-representation above the other birds in that same (como dijo Trini, que la pajarita "Maria" no tenga demasiada sobre-representacion en toda su especie).~~

**Barridos & Grid Search** (Mei)
Implement Early Stopping (xd), de mas aclarar again, para no correr 300 epochs de mas, tirar tiempo, y overfittear while at it. Research if there's some common heuristic for what early-stop margin to set. Replicar lo que pensé para el TP3 de guardarse el valor de loss mas bajo y sus pesos en esa instancia, y si *ese* valor no se mejora despues de $n$ epocas, entonces ahi si se corta el training.

Balance the classes. Use cost-reweighting, applied directly to the PyTorch tensors and to the criterion (cross-entropy). Other methods of balancing are not good for audios and spectrograms. The papers that mention dealing with class imbalance, all use cost re-weighting.

(De mas esta aclarar), try sweeps and grid searches of different architectures, optimization and regularization techniques. Try L2 Reg, try using vs not using ADAM, try varying learning rate or learning rate momentum (todas las 'vistas en la cursada'). Instead of just varying techniques and architectures, also vary some arbitrary values for data-processing.
Try varying the threshold_factor, the segment durations, and maybe even some others? hop_len, nfft, mel_bins? might be too much though.

User cares about final predictions, not cross-entropy loss. Cuando me ponga a correr los barridos y grid search, uso la cross-entropy loss para entrenar y penalizar al modelo, pero para elegir el modelo final para deploy, evaluar el que tenga mejor F1 Score.

**Isolating Samples based on suspected repetition** (Mei)
(Maybe), tratar de poner un max_cap a muestras que sean casi-identicas en {(lat, lon), author}, ya que probablemente representen al mismo pajaro, o at least condiciones de grabacion muy similares. Es un (maybe) igual porque qcyo, tampoco para ponerse *tan* exquisitos, como si tuvieramos 10,000 samples por cada especie. Maybe la unica razon por la que tenemos muestras de Acadian Flycatcher es porque el bueno de Oscar Humberto Marin-Gomez se sentó en el bosque de Quindio, Colombia para grabar algunos pajaros en 2007. Sino lo que diria es marcar una funcion que determine si grabaciones son consideradas is_unique en base a esas condiciones de metadata, y de ahi contar para cada especie, cuantas grabaciones en diff_situation tienen. Si tienen mas que $n_{threshold}$, entonces recortamos a que tengan solo diferentes, o alguna cuenta asi para tampoco sacarles muestras y ahora undersamplearlas (eg: Palomas tienen 2 difs, 100 repetidas le dejan 102; Robins tienen 15 difs les dejas 15). Definitely como primer paso, editar el metadata y agregarle una columna binaria de is_unique. A cada muestra le analizamos is_unique en base a (lat, lon), autor, date. Ask Chat:

- como recomendaria evaluar las diferencias en time
- como recomendaria definir is_unique
- como hacer que recortemos muestras repetidas, pero sin darle overrepresentation a las que tienen pocas unicas y muchas repetidas (dar ejemplo de palomas & robins)
- si hay muchas repetidas, consider elegir la de highest rating

- Add feature to the original metadata, 'recorder'. Being the autor but with numerical IDs instead of names.
- Siempre tener las grabaciones del mismo recorder en el mismo set asi no hay leaking.
- Siempre tener los segmentos del mismo original_audio en el mismo set asi no hay leaking.
- Al CSV final, agregarle 2 features extra para filtrar: 'og_audio', 'recorder' para hacer esto.

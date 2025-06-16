# To-Do

How to add noise to make model more robust? Research what the common pracvtices are, research how the papers did it.

How to make the model deal with noise/silence? Should the model be able to predict if the audio is *not a bird*? What if the model receives just silence? What if it receives some audio that just isn't a bird at all? Shazam says "song not found".

Balance the classes. Use cost-reweighting, applied directly to the PyTorch tensors and to the criterion (cross-entropy). Other methods of balancing are not good for audios and spectrograms. The papers that mention dealing with class imbalance, all use cost re-weighting.

Try limiting the max amount of segments per single audio sample, so that a single very long recording doesn't get significant over-representation above the other birds in that same (como dijo Trini, que la pajarita "Maria" no tenga demasiada sobre-representacion en toda su especie).

(De mas esta aclarar), try sweeps and grid searches of different architectures, optimization and regularization techniques. Try L2 Reg, try using vs not using ADAM, try learning rate momentum (todas las 'vistas en la cursada').

Implement Early Stopping (xd), de mas aclarar again, para no correr 300 epochs de mas, tirar tiempo, y overfittear while at it. Research if there's some common heuristic for what early-stop margin to set. Replicar lo que pensé para el TP3 de guardarse el valor de loss mas bajo y sus pesos en esa instancia, y si *ese* valor no se mejora despues de $n$ epocas, entonces ahi si se corta el training.

User cares about accuracy, not cross-entropy loss. Cuando me ponga a correr los barridos y grid search, uso la cross-entropy loss para entrenar y penalizar al modelo, pero para elegir el modelo final para deploy, evaluar el que tenga mejor accuracy.

(Maybe), tratar de poner un max_cap a muestras que sean casi-identicas en {(lat, lon), author}, ya que probablemente representen al mismo pajaro, o at least condiciones de grabacion muy similares. Es un (maybe) igual porque qcyo, tampoco para ponerse *tan* exquisitos, como si tuvieramos 10,000 samples por cada especie. Maybe la unica razon por la que tenemos muestras de Acadian Flycatcher es porque el bueno de Oscar Humberto Marin-Gomez se sentó en el bosque de Quindio, Colombia para grabar algunos pajaros en 2007. Sino lo que diria es marcar una funcion que determine si grabaciones son consideradas is_unique en base a esas condiciones de metadata, y de ahi contar para cada especie, cuantas grabaciones en diff_situation tienen. Si tienen mas que $n_{threshold}$, entonces recortamos a que tengan solo diferentes, o alguna cuenta asi para tampoco sacarles muestras y ahora undersamplearlas (eg: Palomas tienen 2 difs, 100 repetidas le dejan 102; Robins tienen 15 difs les dejas 15). Definitely como primer paso, editar el metadata y agregarle una columna binaria de is_unique. A cada muestra le analizamos is_unique en base a (lat, lon), autor, date. Ask Chat:

- como recomendaria evaluar las diferencias en time
- como recomendaria definir is_unique
- como hacer que recortemos muestras repetidas, pero sin darle overrepresentation a las que tienen pocas unicas y muchas repetidas (dar ejemplo de palomas & robins)
- si hay muchas repetidas, consider elegir la de highest rating
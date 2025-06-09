# Propuesta de Viabilidad | PF ML

### Fuente de Datos

Los datos se obtuvieron del dataset [BirdClef 2021](https://www.kaggle.com/c/birdclef-2021/data) de Kaggle, extraído de la famosa database de 'xeno-canto' con grabaciones de cantos de aves y sus etiquetas. Encontramos que BirdClef tiene un CSV con metadata de cada grabación, incluyendo entre otras cosas, la dirección del archivo de audio, el nombre científico (y común) del pájaro, y la latitud y longitud donde fue tomada la grabación. El dataset pesa unos 40GB, pero se puede acceder desde Kaggle. Aún asi, se tomo la libertad de descargalo localmente para trabajarlo de manera directa.

### Cantidad de Datos

El dataset de xeno-canto tiene aproximadamente $1,000,000$ de muestras, con mas de $10,000$ clases distintas, se hace imposible un modelo que aprenda todas.

**Formas de solucionarlo:**

- Recortar por regiones (ej: solo Argentina, solo Sudámerica, etc.)
- Recortar por rating (etiqueta existente de calidad de la grabación)
- Recortar especies con muy pocas muestras (o muy pocas muestras de 'alta calidad')

### Balanceo de Datos

Algunas especies tienen muchas mas muestras que otras, podemos usar varios métodos vistos durante la cursada para compensarlo:

- Undersampling
- Oversampling by Duplication
- Oversampling by SMOTE
- Cost Reweighting

### Características de los Datos

**Formato de Audio**
Los audios están en formato .ogg, el cual no es de mucha utilidad para procesar, pero con librosa u otro module, pueden transformarse a .wav para procesar.

**Sample Rate del Audio**
Por lo que se observó por ahora, los audios parecen compartir la misma tasa de muestreo, de 32kHz, sin embargo, si llegaran a haber muestras con distinta sampling rate, se pueden resamplear facilmente con librerias.

**Duración del Audio**
Los audios son de duración variable. Para tratar con esto se pueden tomar ventanas (como es práctica estándar en el procesamiento de audios, y como leímos que se realiza también en los papers adjuntos al formulario).

**Ruido y Silencio en las muestras**
Muchas de las muestras tienen periodos de silencio, o ruido de ambiente sin sonidos de pájaro, lo cual, sobre todo si se aplican windows incorrectas, podría perjudicar al modelo, asociando ruido con una label concreta. Para lidiar con esto, se pueden aprovechar propiedades del audio (también utilizadas por los investigadores citados en los diversos papers) para solo tomar los segmentos donde haya suficiente volumen/energía. O remover las partes con muy poca varianza, que parezcan llanuras en el espectrograma.

### Input de Entrenamiento

El input serán vectores que contienen el valor en escala de grises de cada uno de los píxeles de las imágenes de los espectrogramas (como el caso del TP 3).

### Output del Modelo

Un softmax de N dimensiones (N: cantidad de clases de pájaros), con las probabilidades de que el audio de input pertenezca a cada clase. Y mas especifícamente, la clase de mayor probabilidad (predicción de qué pájaro es).

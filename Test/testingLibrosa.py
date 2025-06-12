import librosa as lb
import numpy as np
import matplotlib.pyplot as plt

audio, sr = lb.load("/home/myopian/Documents/uni/5to sem/ml/ProyectoFinal/ChirpID-backend/Test/audioPrueba.ogg")
print("\nSample rate de audioPrueba =", sr)
print("Duración del audio en segundos =", len(audio) / sr)
ffourier = lb.stft(audio)
S_db = lb.amplitude_to_db(np.abs(ffourier), ref=np.max)

hop=1
rms = lb.feature.rms(S=ffourier, hop_length=hop)  # root-mean-square value for each window, hop-length es la cantidad de frames de audio que se usan para calcular cada valor de energía
print("La energia de audioPrueba en cada ventana de", hop,"frames =", rms)
print("La suma de las energías al cuadrado =", np.sum(rms**2))

plt.figure(figsize=(10, 4))
plt.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (dB)')
plt.xlabel('Frames/samples')
plt.ylabel('Frequency bins')
plt.tight_layout()
plt.show()

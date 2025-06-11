import librosa as lb
import numpy as np
import matplotlib.pyplot as plt

audio, sr = lb.load("audioPrueba.ogg")
print("\nSample rate de audioPrueba =", sr)
print("Duración del audio en segundos =", len(audio) / sr)
ffourier = lb.stft(audio)
S_db = lb.amplitude_to_db(np.abs(ffourier), ref=np.max)

rms = librosa.feature.rms(S=ffourier, hop_length=1)  # root-mean-square value for each frame, hop-length es la cantidad de frames de audio que se usan para calcular cada valor de energía

plt.figure(figsize=(10, 4))
plt.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (dB)')
plt.xlabel('Frames/samples')
plt.ylabel('Frequency bins')
plt.tight_layout()
plt.show()

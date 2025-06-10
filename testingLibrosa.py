import librosa as lb
import numpy as np
import matplotlib.pyplot as plt

audio, sr = lb.load("audioPrueba.ogg")
print("\nSample rate de audioPrueba =", sr)
print("Duración del audio en segundos =", len(audio) / sr)
S = lb.stft(audio)
S_db = lb.amplitude_to_db(np.abs(S), ref=np.max)


plt.figure(figsize=(10, 4))
plt.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (dB)')
plt.xlabel('Frames/samples')
plt.ylabel('Frequency bins')
plt.tight_layout()
plt.show()

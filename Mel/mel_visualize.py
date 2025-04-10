import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# List of audio files
filenames = [
    'genres/jazz/jazz.00001.wav',
    'genres/classical/classical.00001.wav'
]
titles = ['Jazz', 'Classical']

plt.figure(figsize=(12, 6))

for i, filename in enumerate(filenames):
    try:
        # Load and compute Mel Spectrogram
        y, sr = librosa.load(filename, duration=10)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Plot each in a subplot
        plt.subplot(1, 2, i + 1)
        librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'{titles[i]} - Mel Spectrogram')
        plt.tight_layout()
    except Exception as e:
        print(f"⚠️ Failed to process {filename}: {e}")

plt.savefig("mel_spectrogram_comparison.png")

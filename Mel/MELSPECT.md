# 🎵 Mel Spectrogram Overview

## What is a Mel Spectrogram?
A Mel Spectrogram is a visual representation of the frequency content of a sound signal over time, scaled according to the human ear’s perception.

- **X-axis**: Time
- **Y-axis**: Frequency (on the Mel scale)
- **Color**: Energy or amplitude of the signal at a specific time and frequency

## Why Use Mel Spectrograms in Deep Learning?
- Converts raw audio into 2D data (like an image) for CNN models
- Captures both **temporal** and **frequency** information
- More perceptually relevant than linear frequency scale

## Workflow Summary
1. Load audio signal (e.g., using `librosa.load`)
2. Convert to Mel Spectrogram using `librosa.feature.melspectrogram`
3. (Optional) Convert to dB scale using `librosa.power_to_db`
4. Use as input for CNN-based audio classifiers

## Example Visualization
We visualize the Mel Spectrogram using `librosa.display.specshow` and `matplotlib`.

Run `mel_visualize.py` to see the spectrogram for an example audio file.
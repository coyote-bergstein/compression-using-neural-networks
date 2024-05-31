import librosa
import numpy as np
import matplotlib as plt
import soundfile as sf

from constants import *


def load_audio(paths: list, sample_rate: int, duration: float) -> list[np.array]:
    signals = []
    for file_path in paths:
      signal = librosa.load(file_path, sr=sample_rate, duration=duration, mono=True)[0]
      signals.append(signal)
    return signals


def generate_spectrogram(signal: np.array, hop_length: int, frame_size: int) -> np.array:
    S = np.abs(librosa.stft(signal, hop_length=hop_length, n_fft=frame_size, window=WINDOW))
    return S


def show_spectrogram(S: np.array, hop_length: int, frame_size: int, sample_rate: int):
    plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sample_rate, hop_length=hop_length, n_fft=frame_size, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()


def reconstruct_audio(S: np.array, hop_length: int, frame_size: int) -> np.array:
    y_reconstructed = librosa.istft(S, hop_length=hop_length, n_fft=frame_size, window=WINDOW)

    # Normalization
    peak_amplitude = np.max(np.abs(y_reconstructed))
    normalized_y = y_reconstructed / peak_amplitude

    return normalized_y


def save_audio(signal: np.array, output_path: str, sample_rate: int):
    sf.write(output_path, signal, sample_rate)
import librosa
import numpy as np
import matplotlib as plt
import soundfile as sf

from .constants import WINDOW, AUDIO_SAMPLE_RATE


def load_audio(paths: list, sample_rate: int, duration: float) -> list[np.array]:
    signals = []
    for file_path in paths:
        signal = librosa.load(file_path, sr=sample_rate, duration=duration, mono=True)[0]
        signals.append(signal)
    return signals


def generate_spectrogram(signal: np.array, hop_length: int, frame_size: int) -> np.array:
    # S = np.abs(librosa.stft(signal, hop_length=hop_length, n_fft=frame_size, window=WINDOW)) # OLD METHOD
    S = librosa.feature.melspectrogram(y=signal,
                                       sr=AUDIO_SAMPLE_RATE,
                                       n_fft=frame_size,
                                       hop_length=hop_length,
                                       win_length=None,
                                       window=WINDOW,
                                       center=True,
                                       pad_mode='reflect',
                                       power=2.0)
    # n_mels=128)
    S = librosa.power_to_db(S, ref=np.max)
    return S


def show_spectrogram(S: np.array, hop_length: int, frame_size: int, sample_rate: int):
    plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sample_rate, hop_length=hop_length,
                             n_fft=frame_size, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()


def reconstruct_audio(S: np.array, hop_length: int, frame_size: int) -> np.array:
    # y_reconstructed = librosa.istft(S, hop_length=hop_length, n_fft=frame_size, window=WINDOW) # OLD METHOD
    S = librosa.db_to_power(S, ref=1.0)
    y_reconstructed = librosa.feature.inverse.mel_to_audio(S,
                                                           sr=AUDIO_SAMPLE_RATE,
                                                           n_fft=frame_size,
                                                           hop_length=hop_length,
                                                           win_length=None,
                                                           window=WINDOW,
                                                           center=True,
                                                           pad_mode='reflect',
                                                           power=2.0,
                                                           n_iter=32)
    # n_mels=128)

    # Normalization
    peak_amplitude = np.max(np.abs(y_reconstructed))
    normalized_y = y_reconstructed / peak_amplitude

    return normalized_y


def save_audio(signal: np.array, output_path: str, sample_rate: int):
    sf.write(output_path, signal, sample_rate)

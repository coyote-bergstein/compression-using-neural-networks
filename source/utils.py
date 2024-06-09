import librosa
import numpy as np
import matplotlib as plt
import soundfile as sf

def load_audio(paths: list, sample_rate: int, duration: float) -> list[np.array]:
    signals = []
    for file_path in paths:
        signal = librosa.load(file_path, sr=sample_rate, duration=duration, mono=True)[0]
        signals.append(signal)
    return signals

def normalize_standard(audio):
    mean = np.mean(audio)
    std = np.std(audio)
    audio_normalized = (audio - mean) / std
    return audio_normalized, mean, std


def generate_spectrogram(signal: np.array, sr: int, hop_length: int, frame_size: int, win_length: int, window: str, center: bool, pad_mode: str, power: float, n_mels: int) -> np.array:
    # S = np.abs(librosa.stft(signal, hop_length=hop_length, n_fft=frame_size, window=WINDOW)) # OLD METHOD
    S = librosa.feature.melspectrogram(y=signal,
                                       sr=sr,
                                       n_fft=frame_size,
                                       hop_length=hop_length,
                                       win_length=win_length,
                                       window=window,
                                       center=center,
                                       pad_mode=pad_mode,
                                       power=power,
                                       n_mels=n_mels)
    S = librosa.power_to_db(S, ref=np.max)
    return S


def show_spectrogram(S: np.array, hop_length: int, frame_size: int, sample_rate: int):
    plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sample_rate, hop_length=hop_length,
                             n_fft=frame_size, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()


def reconstruct_audio(S: np.array, sr: int, hop_length: int, frame_size: int, mean: float, std: float, win_length: int, window: str, center: bool, pad_mode: str, power: float, n_iter: int) -> np.array:
    # y_reconstructed = librosa.istft(S, hop_length=hop_length, n_fft=frame_size, window=WINDOW) # OLD METHOD
    S = librosa.db_to_power(S, ref=1.0)
    y_reconstructed = librosa.feature.inverse.mel_to_audio(S,
                                                           sr=sr,
                                                           n_fft=frame_size,
                                                           hop_length=hop_length,
                                                           win_length=win_length,
                                                           window=window,
                                                           center=center,
                                                           pad_mode=pad_mode,
                                                           power=power,
                                                           n_iter=n_iter)
    # n_mels=128)
    y_reconstructed = (y_reconstructed * std) + mean
    return y_reconstructed


def save_audio(signal: np.array, output_path: str, sample_rate: int):
    sf.write(output_path, signal, sample_rate)

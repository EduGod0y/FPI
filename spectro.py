import io
import typing as T

import numpy as np
from PIL import Image
import pydub
from scipy.io import wavfile
import torch
import torchaudio


def wav_bytes_from_spectrogram_image(image: Image.Image) -> T.Tuple[io.BytesIO, float]:
    

    #Parametros da transformada inversa:

    max_volume = 50 #Volume máximo do áudio
    power_for_image = 0.25 
    Sxx = spectrogram_from_image(image, max_volume=max_volume, power_for_image=power_for_image)
    
    sample_rate = 44100  # Frequência máxima utilizada [Hz]
    clip_duration_ms = 5000  # Duração do audio [ms]

    bins_per_image = 512 #Número de bins (tamanho)
    n_mels = 512

    
    window_duration_ms = 100  # Duração de cada janela [ms]
    padded_duration_ms = 400  # [ms]
    step_size_ms = 10  # [ms]

    
    num_samples = int(image.width / float(bins_per_image) * clip_duration_ms) * sample_rate
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    samples = waveform_from_spectrogram(
        Sxx=Sxx,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        num_samples=num_samples,
        sample_rate=sample_rate,
        mel_scale=True,
        n_mels=n_mels,
        max_mel_iters=200,
        num_griffin_lim_iters=32,
    )

    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples.astype(np.int16))
    wav_bytes.seek(0)

    duration_s = float(len(samples)) / sample_rate

    return wav_bytes, duration_s


def spectrogram_from_image(
    image: Image.Image, max_volume: float = 50, power_for_image: float = 0.25
) -> np.ndarray:
   
    # Transforma a imagem num array
    data = np.array(image).astype(np.float32)

    # Transforma num array de uma dimensão
    data = data[::-1, :, 0]

    # Inverte
    data = 255 - data

    # Reescalando pelo volume
    data = data * max_volume / 255

    # Invertendo a curva de energia
    data = np.power(data, 1 / power_for_image)

    return data



def waveform_from_spectrogram(
    Sxx: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    num_samples: int,
    sample_rate: int,
    mel_scale: bool = True,
    n_mels: int = 512,
    max_mel_iters: int = 200,
    num_griffin_lim_iters: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    
    Sxx_torch = torch.from_numpy(Sxx).to(device)

    if mel_scale:
        mel_inv_scaler = torchaudio.transforms.InverseMelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
            max_iter=max_mel_iters,
        ).to(device)

        Sxx_torch = mel_inv_scaler(Sxx_torch)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
        n_iter=num_griffin_lim_iters,
    ).to(device)

    waveform = griffin_lim(Sxx_torch).cpu().numpy()

    return waveform


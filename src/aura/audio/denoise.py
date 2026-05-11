import numpy as np
import noisereduce as nr

from aura.config import SAMPLE_RATE

MIN_DENOISE_SAMPLES = 64
DENOISE_PROP_DECREASE = 0.35
SILENCE_RMS_THRESHOLD = 1e-6


def reduce_noise_safely(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Run noisereduce on live buffers without invalid short-window STFT settings."""
    if audio_np.size < MIN_DENOISE_SAMPLES:
        return audio_np
    if float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))) < SILENCE_RMS_THRESHOLD:
        return audio_np

    n_fft = min(1024, int(audio_np.size))
    hop_length = max(1, n_fft // 4)

    return nr.reduce_noise(
        y=audio_np,
        sr=sample_rate,
        stationary=False,
        prop_decrease=DENOISE_PROP_DECREASE,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
    )

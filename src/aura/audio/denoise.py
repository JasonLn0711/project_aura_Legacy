import numpy as np
import noisereduce as nr

from aura.config import SAMPLE_RATE


def reduce_noise_safely(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Run noisereduce on live buffers without invalid short-window STFT settings."""
    if audio_np.size < 64:
        return audio_np

    n_fft = min(1024, int(audio_np.size))
    hop_length = max(1, n_fft // 4)

    return nr.reduce_noise(
        y=audio_np,
        sr=sample_rate,
        stationary=True,
        prop_decrease=0.75,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
    )

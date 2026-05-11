import unittest

import numpy as np
from pydub import AudioSegment

from aura.audio.denoise import reduce_audio_segment_noise, reduce_noise_safely


def snr_db(reference, residual):
    return 10 * np.log10(np.mean(reference**2) / max(np.mean(residual**2), 1e-12))


class DenoiseTests(unittest.TestCase):
    def test_reduce_noise_safely_accepts_short_live_frame(self):
        audio = np.zeros(480, dtype=np.float32)

        output = reduce_noise_safely(audio)

        self.assertEqual(output.shape, audio.shape)
        self.assertEqual(output.dtype, np.float32)
        self.assertIs(output, audio)

    def test_reduce_noise_safely_skips_tiny_frame(self):
        audio = np.zeros(32, dtype=np.float32)

        output = reduce_noise_safely(audio)

        self.assertIs(output, audio)

    def test_reduce_noise_safely_preserves_synthetic_tone(self):
        sample_rate = 16000
        rng = np.random.default_rng(7)
        seconds = 2.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
        clean = (0.12 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        noise = (0.04 * rng.standard_normal(clean.size)).astype(np.float32)
        noisy = (clean + noise).astype(np.float32)

        output = reduce_noise_safely(noisy, sample_rate)

        input_snr = snr_db(clean, noisy - clean)
        output_snr = snr_db(clean, output - clean)
        self.assertFalse(np.isnan(output).any())
        self.assertFalse(np.isinf(output).any())
        self.assertGreaterEqual(output_snr, input_snr - 0.5)

    def test_reduce_audio_segment_noise_preserves_layout(self):
        sample_rate = 16000
        left = np.zeros(sample_rate // 10, dtype=np.int16)
        right = np.zeros(sample_rate // 10, dtype=np.int16)
        stereo = np.stack([left, right], axis=1)
        audio = AudioSegment(
            stereo.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=2,
        )

        output = reduce_audio_segment_noise(audio)

        self.assertEqual(output.frame_rate, sample_rate)
        self.assertEqual(output.sample_width, 2)
        self.assertEqual(output.channels, 2)
        self.assertEqual(len(output), len(audio))


if __name__ == "__main__":
    unittest.main()

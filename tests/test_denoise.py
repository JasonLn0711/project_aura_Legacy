import unittest

import numpy as np

from aura.audio.denoise import reduce_noise_safely


class DenoiseTests(unittest.TestCase):
    def test_reduce_noise_safely_accepts_short_live_frame(self):
        audio = np.zeros(480, dtype=np.float32)

        output = reduce_noise_safely(audio)

        self.assertEqual(output.shape, audio.shape)
        self.assertEqual(output.dtype, np.float32)

    def test_reduce_noise_safely_skips_tiny_frame(self):
        audio = np.zeros(32, dtype=np.float32)

        output = reduce_noise_safely(audio)

        self.assertIs(output, audio)


if __name__ == "__main__":
    unittest.main()

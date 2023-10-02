import os
import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.obsmod.psf import VelocityDispersion

class TestVelocityDispersion(TestBase):
    def test_convolve(self):
        wave = np.linspace(3000, 9000, 6001)
        value1 = np.random.normal(size=wave.shape)
        value2 = np.random.normal(size=wave.shape)
        error = np.full_like(value1, 1.0)
        sigma = np.linspace(3, 5, 6001)

        psf = VelocityDispersion(1000)

        w, v, e, s = psf.convolve(wave, value1, error, size=151, normalize=True)
        self.assertIsInstance(v, np.ndarray)

        w, v, e, s = psf.convolve(wave, [ value1 ], error, size=151, normalize=True)
        self.assertIsInstance(v, list)

        w, v, e, s = psf.convolve(wave, [ value1, value2 ], error, size=151, normalize=True)
        self.assertIsInstance(v, list)
        self.assertEqual(2, len(v))

    def test_get_optimal_size(self):
        psf = VelocityDispersion(300)

        wave = np.linspace(3000, 9000, 6001)
        psf.get_optimal_size(wave, 1e-5)


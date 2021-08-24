import os
import numpy as np
from test.test_base import TestBase

from pfsspec.obsmod.gausspsf import GaussPsf

class TestGaussPsf(TestBase):
    def test_save(self):
        pass

    def test_load(self):
        pass

    def test_get_kernel(self):
        wave = np.linspace(3000, 9000, 6001)
        sigma = np.linspace(3, 5, 6001)

        psf = GaussPsf(wave=wave, sigma=sigma)
        k = psf.get_kernel(4000, np.linspace(-5, 5, 11))
        self.assertTrue(np.all(~np.isnan(k)))

        k = psf.get_kernel(2500, np.linspace(-5, 5, 11))
        self.assertTrue(np.all(~np.isnan(k)))

        k = psf.get_kernel(9500, np.linspace(-5, 5, 11))
        self.assertTrue(np.all(~np.isnan(k)))

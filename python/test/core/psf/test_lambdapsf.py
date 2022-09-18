import os
import numpy as np

from test.core import TestBase
from pfs.ga.pfsspec.core.util.math import *
from pfs.ga.pfsspec.core.psf import LambdaPsf

class TestLambdaPsf(TestBase):
        
    def test_eval_kernel(self):
        wave = np.linspace(3000, 9000, 6001)
        
        psf = LambdaPsf(lambda w, dw: gauss(dw, s=w / 1000.0))
        
        w, dw, k, idx, shift = psf.eval_kernel(wave, size=11)
        self.assertEqual(k.shape, (5991, 11))

    def test_convolve(self):
        wave = np.linspace(3000, 9000, 6001)
        value = np.random.normal(size=wave.shape)
        error = np.full_like(value, 1.0)

        psf = LambdaPsf(lambda w, dw: gauss(dw, s=w / 1000.0))
        w, v, e, s = psf.convolve(wave, value, error)
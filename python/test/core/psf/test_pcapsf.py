import os
import numpy as np

from test.core import TestBase
from pfs.ga.pfsspec.core.psf import GaussPsf, PcaPsf

class TestPcaPsf(TestBase):
    def test_save(self):
        pass

    def test_load(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/import/mr/pca.h5')
        psf = PcaPsf()
        psf.load(filename)
        
    def test_from_psf(self):
        wave = np.linspace(3000, 9000, 6001)
        sigma = np.linspace(3, 5, 6001)

        g = GaussPsf(wave=wave, sigma=sigma)

        psf = PcaPsf.from_psf(g, wave, 11, normalize=True, truncate=4)

    def test_eval_kernel(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/import/r/pca.h5')
        psf = PcaPsf()
        psf.load(filename)

        k, idx, shift = psf.eval_kernel(psf.wave)

    def test_eval_kernel_from_psf(self):
        wave = np.linspace(3000, 9000, 6001)
        sigma = np.linspace(3, 5, 6001)

        g = GaussPsf(wave=wave, sigma=sigma)

        psf = PcaPsf.from_psf(g, wave, 11, normalize=True, truncate=4)

        k, idx, shift = psf.eval_kernel(wave, normalize=True)

    def test_convolve(self):
        wave = np.linspace(3000, 9000, 6001)
        value1 = np.random.normal(size=wave.shape)
        value2 = np.random.normal(size=wave.shape)
        error1 = np.full_like(value1, 1.0)
        error2 = np.full_like(value2, 1.0)
        sigma = np.linspace(3, 5, 6001)

        g = GaussPsf(wave=wave, sigma=sigma)
        psf = PcaPsf.from_psf(g, wave, 11, normalize=True, truncate=4)

        w, v, e, s = psf.convolve(wave, value1, error1)
        self.assertIsInstance(v, np.ndarray)

        w, v, e, s = psf.convolve(wave, [ value1 ], error1)
        self.assertIsInstance(v, list)

        w, v, e, s = psf.convolve(wave, [ value1, value2 ], [ error1, error2 ])
        self.assertIsInstance(v, list)
        self.assertEqual(2, len(v))
        

        
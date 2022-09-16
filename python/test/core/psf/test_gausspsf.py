import os
import numpy as np

from test.core import TestBase
from pfs.ga.pfsspec.core.psf import GaussPsf

class TestGaussPsf(TestBase):
    def test_save(self):
        pass

    def test_load(self):
        pass

    def test_eval_kernel_at(self):
        wave = np.linspace(3000, 9000, 6001)
        sigma = np.linspace(3, 5, 6001)

        psf = GaussPsf(wave=wave, sigma=sigma)

        k = psf.eval_kernel_at(np.array([4000]), dwave=np.linspace(-5, 5, 11))
        self.assertTrue(np.all(~np.isnan(k)))

        k = psf.eval_kernel_at(np.array([4000]), dwave=np.linspace(-5, 5, 11), normalize=True)
        self.assertTrue(np.all(~np.isnan(k)))

        k = psf.eval_kernel_at(np.array([2500]), dwave=np.linspace(-5, 5, 11))
        self.assertTrue(np.all(~np.isnan(k)))

        k = psf.eval_kernel_at(np.array([2500]), dwave=np.linspace(-5, 5, 11), normalize=True)
        self.assertTrue(np.all(~np.isnan(k)))

        k = psf.eval_kernel_at(np.array([9500]), dwave=np.linspace(-5, 5, 11))
        self.assertTrue(np.all(~np.isnan(k)))

    def test_eval_kernel(self):
        wave = np.linspace(3000, 9000, 6001)
        sigma = np.linspace(3, 5, 6001)

        psf = GaussPsf(wave=wave, sigma=sigma)

        w, k, idx, shift = psf.eval_kernel(wave, 11, normalize=True)
        self.assertEqual((5991, 11), k.shape)
        self.assertEqual((5991, 11), idx.shape)
        self.assertEqual(5, shift)

        w, k, idx, shift = psf.eval_kernel(wave, 11, s=np.s_[::10], normalize=True)
        self.assertEqual((600, 11), k.shape)
        self.assertEqual((600, 11), idx.shape)
        self.assertEqual(5, shift)

    def test_convolve(self):
        wave = np.linspace(3000, 9000, 6001)
        value1 = np.random.normal(size=wave.shape)
        value2 = np.random.normal(size=wave.shape)
        error = np.full_like(value1, 1.0)
        sigma = np.linspace(3, 5, 6001)

        psf = GaussPsf(wave=wave, sigma=sigma)

        w, v, e, s = psf.convolve(wave, value1, error)
        self.assertIsInstance(v, np.ndarray)

        w, v, e, s = psf.convolve(wave, [ value1 ], error)
        self.assertIsInstance(v, list)

        w, v, e, s = psf.convolve(wave, [ value1, value2 ], error)
        self.assertIsInstance(v, list)
        self.assertEqual(2, len(v))

        
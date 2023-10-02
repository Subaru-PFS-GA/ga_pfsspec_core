import os
import numpy as np
import pandas as pd

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.obsmod.psf import TabulatedPsf, GaussPsf

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
        dwave = np.stack(wave.size * [ np.arange(-5, 6) ], axis=0)
        sigma = np.linspace(3, 5, 6001)

        psf = GaussPsf(wave=wave, sigma=sigma)

        # Specify size

        w, dw, k, idx, shift = psf.eval_kernel(wave, size=11, normalize=True)
        self.assertEqual((5991, 11), k.shape)
        self.assertEqual((5991, 11), idx.shape)
        self.assertEqual(5, shift)

        w, dw, k, idx, shift = psf.eval_kernel(wave, size=11, s=np.s_[::10], normalize=True)
        self.assertEqual((600, 11), k.shape)
        self.assertEqual((600, 11), idx.shape)
        self.assertEqual(5, shift)

        # Specify dwave

        w, dw, k, idx, shift = psf.eval_kernel(wave, dwave=dwave, normalize=True)
        self.assertEqual((6001, 11), k.shape)
        self.assertIsNone(idx)
        self.assertEqual(0, shift)

        w, dw, k, idx, shift = psf.eval_kernel(wave, dwave=dwave, s=np.s_[::10], normalize=True)
        self.assertEqual((601, 11), k.shape)
        self.assertIsNone(idx)
        self.assertEqual(0, shift)

    def test_convolve(self):
        wave = np.linspace(3000, 9000, 6001)
        value1 = np.random.normal(size=wave.shape)
        value2 = np.random.normal(size=wave.shape)
        error = np.full_like(value1, 1.0)
        sigma = np.linspace(3, 5, 6001)

        psf = GaussPsf(wave=wave, sigma=sigma)
        size = psf.get_optimal_size(wave)

        w, v, e, s = psf.convolve(wave, value1, error, size=size)
        self.assertIsInstance(v, np.ndarray)

        w, v, e, s = psf.convolve(wave, [ value1 ], error, size=size)
        self.assertIsInstance(v, list)

        w, v, e, s = psf.convolve(wave, [ value1, value2 ], error, size=size)
        self.assertIsInstance(v, list)
        self.assertEqual(2, len(v))

    def test_from_psf(self):
        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/etc/mr/psf.dat')
        df = pd.read_csv(fn, delimiter='\s+', header=None, names=['arm', 'pix', 'lam',] + [ f'psf_{i}' for i in range(31)])
        df = df[df['arm'] == 3]
        k = np.array(df.iloc[:, 3:])
        size = k.shape[-1]
        w = np.array(df['lam'] * 10)
        dw = w[:size] - w[size // 2]

        tab_psf = TabulatedPsf()
        tab_psf.wave = w
        tab_psf.dwave = w[:, np.newaxis] + dw
        tab_psf.kernel = k

        gauss_psf = GaussPsf.from_psf(tab_psf, w, dwave=dw)

        value1 = np.random.normal(size=w.shape)
        w, v, e, s = gauss_psf.convolve(w, value1)

    def test_get_optimal_size(self):
        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/import/mr.2/gauss.h5')
        psf = GaussPsf()
        psf.load(fn)

        wave = np.linspace(3000.0032915987117, 14999.886673223433, 160944)

        size = psf.get_optimal_size(wave)

    def test_get_width(self):
        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/import/mr.2/gauss.h5')
        psf = GaussPsf()
        psf.load(fn)

        wave = np.linspace(3000.0032915987117, 14999.886673223433, 160944)

        width = psf.get_width(wave)
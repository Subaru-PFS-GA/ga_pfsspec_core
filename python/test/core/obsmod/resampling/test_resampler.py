import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.obsmod.resampling.resampler import Resampler

class TestResampler(TestBase):
    def test_find_edges(self):
        res = Resampler()

        wave_edges = np.linspace(3000, 9000, 6001)
        wave = 0.5 * (wave_edges[1:] + wave_edges[:-1])
        we = res.find_edges(wave)

        self.assertEqual(we.shape, wave_edges.shape)

    def test_resample_error(self):
        res = Resampler()

        wave_edges = np.linspace(3000, 9000, 6001)
        wave = 0.5 * (wave_edges[1:] + wave_edges[:-1])
        flux = np.full_like(wave, 1.0)
        sigma = np.linspace(1, 3, 6000)

        nwave_edges = np.linspace(3000, 9000, 601)
        nwave = 0.5 * (nwave_edges[1:] + nwave_edges[:-1])

        res.init(nwave, nwave_edges)
        nsigma = res.resample_error(wave, wave_edges, flux, sigma)
        res.reset()

    def test_resample_mask(self):
        res = Resampler()

        wave_edges = np.linspace(3000, 9000, 6001)
        wave = 0.5 * (wave_edges[1:] + wave_edges[:-1])
        mask = (np.random.uniform(size=wave.shape) < 0.5)

        nwave_edges = np.linspace(3000, 9000, 601)
        nwave = 0.5 * (nwave_edges[1:] + nwave_edges[:-1])

        res.init(nwave, nwave_edges)
        nmask = res.resample_mask(wave, wave_edges, mask)
        res.reset()
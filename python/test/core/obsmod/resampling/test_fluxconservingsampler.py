import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core import Spectrum
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler

class TestFluxConservingResampler(TestBase):

    def test_resample(self):
        spec = Spectrum()
        spec.wave_edges = np.linspace(3000, 9000, 6001)
        spec.wave = 0.5 * (spec.wave_edges[1:] + spec.wave_edges[:-1])
        spec.flux = np.random.uniform(0, 1, size=spec.wave.shape)
        spec.flux_err = np.linspace(3, 5, spec.wave.shape[0])
        
        nwave_edges = spec.wave_edges[::5]
        nwave = 0.5 * (nwave_edges[1:] + nwave_edges[:-1])

        res = FluxConservingResampler()
        spec.apply_resampler(res, nwave, nwave_edges)

    def test_resample_value(self):
        wave_edges = np.linspace(3000, 9000, 6001)
        wave = 0.5 * (wave_edges[1:] + wave_edges[:-1])

        nwave_edges = wave_edges[::5]
        nwave = 0.5 * (nwave_edges[1:] + nwave_edges[:-1])

        value = np.random.uniform(0, 1, size=wave.shape)
        sigma = np.linspace(3, 5, wave.shape[0])

        res = FluxConservingResampler()
        res.init(nwave, nwave_edges)
        nvalue, nsigma = res.resample_value(wave, wave_edges, value, sigma)
        res.reset()

        self.assertEqual(nvalue.shape, nwave.shape)
        self.assertEqual(nsigma.shape, nwave.shape)
        
    def test_resample_error(self):
        pass

    def test_resample_mask(self):
        pass
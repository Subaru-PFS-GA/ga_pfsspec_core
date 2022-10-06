import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core import Spectrum
from pfs.ga.pfsspec.core import Pipeline
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler

class TestPipeline(TestBase):

    def get_spectrum(self) -> Spectrum:
        p = Pipeline()
        p.wave = [3000, 9000]
        p.wave_bins = 3000
        p.wave_log = True
        
        s = Spectrum()
        s.wave, s.wave_edges = p.get_wave_const()
        s.flux = np.random.uniform(0.8, 1.2, size=s.wave.shape)
        
        return s

    def test_is_constant_wave(self):
        pass

    def test_get_wave(self):
        pass

    def test_get_wave_const(self):
        # Only one of lin and log can be True
        p = Pipeline()
        p.wave_lin = p.wave_log = True
        try:
            p.get_wave_const()
            self.fail()
        except:
            pass

        # When lin or log, wave and bins must be set
        p = Pipeline()
        p.wave_lin = True
        try:
            p.get_wave_const()
            self.fail()
        except:
            pass

        # When bins is defines, lin or log must be True
        p = Pipeline()
        p.wave_bins = 2000
        try:
            p.get_wave_const()
            self.fail()
        except:
            pass

        p = Pipeline()
        p.wave = [3000, 9000]
        p.wave_bins = 3000
        p.wave_lin = True
        p.wave_log = False

        centers, edges = p.get_wave_const()
        self.assertEqual((3000,), centers.shape)
        self.assertEqual((2, 3000), edges.shape)

    def test_get_wave_spec(self):
        pass

    def test_wave_count(self):
        pass

    def test_run_step_restframe(self):
        p = Pipeline()
        p.restframe = True

        # Run with properly initialized spectrum
        s = self.get_spectrum()
        s.redshift = -0.001
        p.run_step_restframe(s)
        self.assertEqual(0.0, s.redshift)

        # Run with missing wave_edges
        s = self.get_spectrum()
        s.wave_edges = None
        s.redshift = -0.001
        p.run_step_restframe(s)
        self.assertEqual(0.0, s.redshift)

        # Run with no known redshift
        s = Spectrum()
        s.redshift = None
        try:
            p.run_step_restframe(s)
            self.fail()
        except ValueError:
            pass

    def test_run_step_redshift(self):
        p = Pipeline()
        p.redshift = 0.001

        # Starting from redshift 0
        s = self.get_spectrum()
        p.run_step_redshift(s)
        self.assertEqual(0.001, s.redshift)

        s = self.get_spectrum()
        p.run_step_redshift(s, redshift=0.002)
        self.assertEqual(0.002, s.redshift)

        # Partially initialized spectrum
        s = self.get_spectrum()
        s.wave_edges = None
        p.run_step_redshift(s)
        self.assertEqual(0.001, s.redshift)

        # Starting from non-zero redshift
        s = self.get_spectrum()
        s.redshift = 0.001
        try:
            p.run_step_redshift(s)
            self.fail()
        except ValueError:
            pass

    def test_run_step_resample(self):
        p = Pipeline()
        p.wave_resampler = FluxConservingResampler()
        w = np.linspace(3500, 7500, 1001)
        p.wave_edges = np.stack([w[:-1], w[1:]])
        p.wave = 0.5 * (p.wave_edges[1:] + p.wave_edges[:-1])
        s = self.get_spectrum()
        p.run_step_resample(s)

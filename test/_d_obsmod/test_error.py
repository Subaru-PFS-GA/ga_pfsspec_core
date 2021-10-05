import os
import numpy as np
from test.test_base import TestBase

from pfsspec.obsmod.observation import Observation

class TestError(TestBase):
    def test_read(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/noise/sim8hr.dat')
        error = Observation()
        error.read(filename)
        error.plot()

        self.assertEqual((12288,), error.wave.shape)
        self.assertEqual((12288,), error.sigma.shape)

        self.save_fig()

    def test_resample(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/noise/sim8hr.dat')
        error = Observation()
        error.read(filename)
        error.plot()

        nwave = np.linspace(3810, 12590, 3600)
        error.resample(nwave)

        self.assertEqual((3600,), error.wave.shape)
        self.assertEqual((3600,), error.sigma.shape)

        self.save_fig()
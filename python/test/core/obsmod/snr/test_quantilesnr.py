import numpy as np

from pfs.ga.pfsspec.core.obsmod.snr import QuantileSnr

from ...test_base import TestBase

class QuantileSnrTest(TestBase):

    def test_get_snr(self):
        v = np.linspace(1, 5, 100)
        s = np.full_like(v, 0.3)
        snr = QuantileSnr().get_snr(v, s)
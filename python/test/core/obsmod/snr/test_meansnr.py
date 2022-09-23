import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.obsmod.snr import MeanSnr

class MeanSnrTest(TestBase):

    def test_get_snr(self):
        v = np.linspace(1, 5, 100)
        s = np.full_like(v, 0.3)
        snr = MeanSnr().get_snr(v, s)
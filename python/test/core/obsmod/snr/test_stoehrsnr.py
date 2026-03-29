import numpy as np

from pfs.ga.pfsspec.core.obsmod.snr import StoehrSnr

from ...test_base import TestBase

class StoehrSnrTest(TestBase):

    def test_get_snr(self):
        v = np.linspace(1, 5, 100)
        snr = StoehrSnr().get_snr(v)
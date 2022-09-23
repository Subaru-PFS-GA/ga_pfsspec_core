import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.obsmod.snr import StoehrSnr

class StoehrSnrTest(TestBase):

    def test_get_snr(self):
        v = np.linspace(1, 5, 100)
        snr = StoehrSnr().get_snr(v)
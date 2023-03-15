import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.sampling import UniformDistribution

class MeanSnrTest(TestBase):

    def test_init_from_args(self):
        d = UniformDistribution()
        d.init_from_args([0, 1])

    def test_sample(self):
        d = UniformDistribution()
        d.init_from_args([0, 1])
        s = d.sample(size=(10,))
        self.assertEqual((10,), s.shape)
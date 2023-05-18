import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.sampling import BetaDistribution

class BetaDistributionTest(TestBase):

    def test_init_from_args(self):
        d = BetaDistribution()
        d.init_from_args([0.9, 0.8])

        d = BetaDistribution()
        d.init_from_args([0.9, 0.8, 1, 2])

        d = BetaDistribution()
        d.init_from_args([1, 1, 0.5, 0.6, -1, 1])

    def test_sample(self):
        d = BetaDistribution()
        d.init_from_args([0.9, 0.8])
        s = d.sample(size=(10,))
        self.assertEqual((10,), s.shape)

        d = BetaDistribution()
        d.init_from_args([1, 1, 0.5, 0.6, -1, 1])
        s = d.sample(size=(10,))
        self.assertEqual((10,), s.shape)

    def test_pdf(self):
        d = BetaDistribution()
        d.init_from_args([0.9, 0.8])
        s = d.sample(size=(10,))
        p = d.pdf(s)
        self.assertEqual((10,), p.shape)

        d = BetaDistribution()
        d.init_from_args([1, 1, 0.5, 0.6, -1, 1])
        s = d.sample(size=(10,))
        p = d.pdf(s)
        self.assertEqual((10,), p.shape)
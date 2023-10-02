import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.sampling import MultivariateNormalDistribution

class MultivariateNormalDistributionTest(TestBase):

    def test_init_from_args(self):
        d = MultivariateNormalDistribution()
        d.init_from_args([0, 1])

        d = MultivariateNormalDistribution()
        d.init_from_args([0, 1, -1, 1, 1, -1])

        d = MultivariateNormalDistribution()
        d.init_from_args([0, 1, -1, 1, 1, -1, 0, 1, -1, 1, 1, -1])

    def test_sample(self):
        d = MultivariateNormalDistribution()
        d.init_from_args([0, 1])
        s = d.sample(size=(10,))
        self.assertEqual((10,), s.shape)

        d = MultivariateNormalDistribution()
        d.init_from_args([0, 1, 1, 0, 0, 1])
        s = d.sample(size=(10,))
        self.assertEqual((10, 2), s.shape)

    def test_pdf(self):
        d = MultivariateNormalDistribution()
        d.init_from_args([0, 1])
        s = d.sample(size=(10,))
        p = d.pdf(s)
        self.assertEqual((10,), p.shape)

        d = MultivariateNormalDistribution()
        d.init_from_args([0, 1, 1, 0, 0, 1])
        s = d.sample(size=(10))
        p = d.pdf(s)
        self.assertEqual((10,), p.shape)
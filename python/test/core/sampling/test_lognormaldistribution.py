import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.sampling import LogNormalDistribution

class LogNormalDistributionTest(TestBase):

    def test_init_from_args(self):
        d = LogNormalDistribution()
        d.init_from_args([1])

        d = LogNormalDistribution()
        d.init_from_args([1, -1, 1])

        d = LogNormalDistribution()
        d.init_from_args([1, 1, 2, -1, 1])

    def test_sample(self):
        d = LogNormalDistribution()
        d.init_from_args([0.9])
        s = d.sample(size=(10,))
        self.assertEqual((10,), s.shape)

        d = LogNormalDistribution()
        d.init_from_args([0.9, -1, 2, 1.2, 1.5])
        s = d.sample(size=(10,))
        self.assertEqual((10,), s.shape)

    def test_pdf(self):
        d = LogNormalDistribution()
        d.init_from_args([0.9])
        s = d.sample(size=(10,))
        p = d.pdf(s)
        self.assertEqual((10,), p.shape)

        d = LogNormalDistribution()
        d.init_from_args([0.9, -1, 2, 1.2, 1.5])
        s = d.sample(size=(10,))
        p = d.pdf(s)
        self.assertEqual((10,), p.shape)
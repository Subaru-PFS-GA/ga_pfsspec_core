import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.sampling import NormalDistribution

class NormalDistributionTest(TestBase):

    def test_init_from_args(self):
        d = NormalDistribution()
        d.init_from_args([0, 1])

        d = NormalDistribution()
        d.init_from_args([0, 1, -1, 1])

    def test_sample(self):
        d = NormalDistribution()
        d.init_from_args([0, 1])
        s = d.sample(size=(10,))
        self.assertEqual((10,), s.shape)

        d = NormalDistribution()
        d.init_from_args([0, 1, -1, 1])
        s = d.sample(size=(10,))
        self.assertEqual((10,), s.shape)

    def test_pdf(self):
        d = NormalDistribution()
        d.init_from_args([0, 1])
        s = d.sample(size=(10,))
        p = d.pdf(s)
        self.assertEqual((10,), p.shape)

        d = NormalDistribution()
        d.init_from_args([0, 1, -1, 1])
        s = d.sample(size=(10,))
        p = d.pdf(s)
        self.assertEqual((10,), p.shape)
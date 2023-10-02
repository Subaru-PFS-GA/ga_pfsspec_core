import numpy as np
import numpy.testing as npt

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.sampling import CompositeDistribution
from pfs.ga.pfsspec.core.sampling import UniformDistribution, NormalDistribution

class CompositeDistributionTest(TestBase):

    def test_init_from_args(self):
        d = CompositeDistribution()
        d.init_from_args([2, 8,
                          '[', 'uniform', -1, 1, ']',
                          '[', 'normal', 0, 2, -1, -1, ']'
                          ])
        
        npt.assert_almost_equal(np.array([0.2, 0.8]), d.weights)
        self.assertIsInstance(d.dists[0], UniformDistribution)
        self.assertIsInstance(d.dists[1], NormalDistribution)

    def test_sample(self):
        d = CompositeDistribution()
        d.init_from_args([2, 8,
                          '[', 'uniform', 2, 3, ']',
                          '[', 'normal', 0, 2, -1, 1, ']'
                          ])
        s = d.sample(size=(10,))
        self.assertEqual((10,), s.shape)
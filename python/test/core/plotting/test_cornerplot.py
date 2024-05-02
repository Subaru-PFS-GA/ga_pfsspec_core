import os
import numpy as np
import matplotlib.pyplot as plt

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.plotting import DiagramPage, DiagramAxis, CornerPlot

class CornerPlotTest(TestBase):

    def test_errorbar(self):
        f = DiagramPage(3, 3)
        cc = CornerPlot(f, 
                        [
                            DiagramAxis(label='x'),
                            DiagramAxis(label='y'),
                            DiagramAxis(label='z'),
                        ])

        cc.errorbar(([1], [1], [0.5]), ([2], [1], [0.5]), ([3], [0.5]), fmt='o')

        for i in range(3):
            cc.update_limits(i, (-5, 5))
        cc.apply()

        f.save(os.path.expandvars(os.path.join('{$PFSSPEC_TEST}', self.get_test_filename(ext='.png'))))

    def test_covariance(self):
        f = DiagramPage(3, 3)
        cc = CornerPlot(f, 
                        [
                            DiagramAxis(label='x'),
                            DiagramAxis(label='y'),
                            DiagramAxis(label='z'),
                        ])

        mu = np.array([0, 1, 2])
        s = np.array([1.0, 1.5, 2.0])
        corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
        cov = corr * np.einsum('i,j->ij', s, s)
        
        cc.covariance(mu, cov, sigma=1)
        cc.covariance(mu, cov, sigma=2, ls='--')
        cc.covariance(mu, cov, sigma=3, ls=':')

        cc.errorbar((0, 1.0), (1, 1.5), (2, 2.0), fmt='o')

        for i in range(3):
            cc.update_limits(i, (-5, 5))
        cc.apply()

        f.save(os.path.expandvars(os.path.join('{$PFSSPEC_TEST}', self.get_test_filename(ext='.png'))))
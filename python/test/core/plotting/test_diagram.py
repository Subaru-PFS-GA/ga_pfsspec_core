import os
import numpy as np
import matplotlib.pyplot as plt

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.plotting import Diagram

class DiagramTest(TestBase):

    def test_errorbar(self):
        f, ax = plt.subplots(1, 1)
        d = Diagram(ax=ax)

        d.errorbar(ax, 0, 0, fmt='o')
        d.errorbar(ax, 1, 1, 1, 1, fmt='o')
        d.errorbar(ax, [2], [2], [[1], [0.5]], [[1], [0.5]], fmt='o')

        d.update_limits(0, (-5, 5))
        d.update_limits(1, (-5, 5))
        d.apply()

        f.savefig(os.path.expandvars(os.path.join('{$PFSSPEC_TEST}', self.get_test_filename(ext='.png'))))
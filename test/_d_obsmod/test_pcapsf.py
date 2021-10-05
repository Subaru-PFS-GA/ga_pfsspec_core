import os
import numpy as np
from test.test_base import TestBase

from pfsspec.obsmod.pcapsf import PcaPsf

class TestGaussPsf(TestBase):
    def test_save(self):
        pass

    def test_load(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/mr/pca.h5')
        psf = PcaPsf()
        psf.load(filename)

    def test_get_kernel(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/r/pca.h5')
        psf = PcaPsf()
        psf.load(filename)

        k = psf.get_kernel(4000)
        
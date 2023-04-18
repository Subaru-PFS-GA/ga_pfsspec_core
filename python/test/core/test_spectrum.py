import numpy as np
import numpy.testing as npt

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core import Spectrum

class TestSpectrum(TestBase):
    # TODO: implement stellar library agnostic tests of basic function

    def test_get_mask_as_bool(self):
        wave = np.linspace(3000, 9000, 6000)
        
        # None
        m = Spectrum.get_mask_as_bool(wave, None)
        self.assertIsNone(m)

        # Slice
        mask = np.s_[100:-100]
        m = Spectrum.get_mask_as_bool(wave, mask)
        self.assertEqual(m.sum(), 5800)

        # Boolean
        mask = np.random.normal(size=wave.shape) > 0
        m = Spectrum.get_mask_as_bool(wave, mask)
        npt.assert_equal(mask, m)

        # Integer
        mask = np.random.randint(0, 1, size=wave.shape)
        m = Spectrum.get_mask_as_bool(wave, mask)

    def test_get_mask_as_int(self):
        wave = np.linspace(3000, 9000, 6000)

        # None
        m = Spectrum.get_mask_as_int(wave, None)
        self.assertIsNone(m)

        # Slice
        mask = np.s_[100:-100]
        m = Spectrum.get_mask_as_int(wave, mask, bits=1)
        self.assertEqual(m.sum(), 200)

        # Boolean
        mask = np.random.normal(size=wave.shape) > 0
        m = Spectrum.get_mask_as_int(wave, mask, bits=1)
        self.assertEqual(mask.sum(), 6000 - m.sum())

        # Integer
        mask = np.full(wave.shape, 1, dtype=int)
        m = Spectrum.get_mask_as_int(wave, mask, bits=1)
        npt.assert_equal(mask, m)
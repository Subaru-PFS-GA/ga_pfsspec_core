from test.test_base import TestBase
import os
from astropy.io import fits

from pfsspec.surveys.sdss1spectrumreader import Sdss1SpectrumReader

class TestSdss1SpectrumReader(TestBase):
    def test_read(self):
        filename = Sdss1SpectrumReader.get_filename(52000, 288, 97)
        filename = os.path.join(self.PFSSPEC_SDSS_DATA_PATH, filename)
        with fits.open(filename) as hdus:
            reader = Sdss1SpectrumReader(hdus)
            spec = reader.read()

    def test_get_filename(self):
        filename = Sdss1SpectrumReader.get_filename(52000, 288, 97)
        self.assertEqual('das2/spectro/1d_26/0288/1d/spSpec-52000-0288-097.fit', filename)
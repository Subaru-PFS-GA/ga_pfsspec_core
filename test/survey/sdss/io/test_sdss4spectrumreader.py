from test.test_base import TestBase
import os
from astropy.io import fits

from pfsspec.surveys.sdss4spectrumreader import Sdss4SpectrumReader

class TestSdss4SpectrumReader(TestBase):
    def test_read(self):
        raise NotImplementedError()
        # filename = Sdss1SpectrumReader.get_filename(52000, 288, 97)
        # filename = os.path.join(self.PFSSPEC_SDSS_DATA_PATH, filename)
        # with fits.open(filename) as hdus:
        #     reader = Sdss1SpectrumReader(hdus)
        #     spec = reader.read()

    def test_get_filename(self):
        filename = Sdss4SpectrumReader.get_filename(52000, 288, 97)
        self.assertEqual('sas/dr16/sdss/spectro/redux/26/spectra/lite/0288/spec-0288-52000-0097.fits', filename)
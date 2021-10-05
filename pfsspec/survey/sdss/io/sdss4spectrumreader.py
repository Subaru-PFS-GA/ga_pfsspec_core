import os
import sys
import numpy as np
import requests
from astropy.io import fits

from pfsspec.data.spectrumreader import SpectrumReader

class Sdss4SpectrumReader(SpectrumReader):
    """
    Reads SDSS 4 (also BOSS) spectra
    """

    # https://data.sdss.org/datamodel/files/SPECTRO_REDUX/RUN2D/spectra/lite/PLATE4/spec.html
    # https://data.sdss.org/datamodel/files/SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html
    # https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html

    def __init__(self, orig=None):
        super(Sdss4SpectrumReader, self).__init__(orig=orig)

        if isinstance(orig, Sdss4SpectrumReader):
            self.path = orig.path
        else:
            self.path = None

    def create_spectrum(self, hdus):
        # TODO: we could figure out spectrum type here from the FITS header
        raise NotImplementedError()

    def read(self, hdus):
        wave = 10**hdus[1].data['loglam']
        flux = hdus[1].data['flux']
        ivar = hdus[1].data['ivar']
        flux_err = np.sqrt(1 / np.where(ivar == 0, np.inf, ivar))
        flux_sky = hdus[1].data['sky']        
        mask = hdus[1].data['or_mask']

        spec = self.create_spectrum(hdus)
        spec.wave = wave
        spec.flux = flux
        spec.flux_err = flux_err
        spec.flux_sky = flux_sky
        spec.mask = mask
        return spec

    def read_header(self, hdus, row, spec):
        spec.id = row['id']
        spec.mjd = row['mjd']
        spec.plate = row['plate']
        spec.fiber = row['fiber']

    @staticmethod
    def get_filename(mjd, plate, fiber, dr='dr16', run2d='26'):
        # sas/dr16/sdss/spectro/redux/26/spectra/lite/0266/spec-0266-51602-0002.fits
        return 'sas/{:s}/sdss/spectro/redux/{:s}/spectra/lite/{:04d}/spec-{:04d}-{:5d}-{:04d}.fits'.format(
            dr, run2d, int(plate), int(plate), int(mjd), int(fiber))

    def load_spectrum(self, index, row):
        filename = Sdss4SpectrumReader.get_filename(row['mjd'], row['plate'], row['fiber'])
        filename = os.path.join(self.path, filename)
        
        # If the file is not available at a local/lan path, download from the web
        if not os.path.isfile(filename):
            url = row['url']
            req = requests.get(url, allow_redirects=True)
            
            dir, _ = os.path.split(os.path.abspath(filename))
            os.makedirs(dir, exist_ok=True)
            
            with open(filename, 'wb') as f:
                f.write(req.content)

        with fits.open(filename, memmap=False) as hdus:
            spec = self.read(hdus)
            spec.index = index
            self.read_header(hdus, row, spec)
            return spec

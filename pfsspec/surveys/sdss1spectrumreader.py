import os
import sys
import numpy as np
from astropy.io import fits

from pfsspec.data.spectrumreader import SpectrumReader

class Sdss1SpectrumReader(SpectrumReader):
    def __init__(self, orig=None):
        super(Sdss1SpectrumReader, self).__init__(orig=orig)

        if isinstance(orig, Sdss1SpectrumReader):
            self.path = orig.path
        else:
            self.path = None

    def create_spectrum(self, hdus):
        # TODO: we could figure out spectrum type here from the FITS header
        raise NotImplementedError()

    def read(self, file):
        loglambda0 = file[0].header['COEFF0  ']
        loglambda1 = file[0].header['COEFF1  ']
        numbins = file[0].data.shape[1]
        logwave = loglambda0 + loglambda1 * np.arange(0, numbins)

        spec = self.create_spectrum(file)
        spec.wave = wave = 10 ** logwave
        spec.flux = file[0].data[0, :]
        spec.flux_err = file[0].data[2, :]
        spec.flux_sky = file[0].data[4, :]
        spec.mask = np.int32(file[0].data[3, :])
        return spec

    def read_header(self, hdus, row, spec):
        spec.id = row['id']
        spec.mjd = row['mjd']
        spec.plate = row['plate']
        spec.fiber = row['fiber']

    @staticmethod
    def get_filename(mjd, plate, fiber, das='das2', ver='1d_26'):
        # .../das2/spectro/1d_26/0288/1d/spSpec-52000-0288-005.fit
        return '{:s}/spectro/{:s}/{:04d}/1d/spSpec-{:5d}-{:04d}-{:03d}.fit'.format(das, ver, int(plate), int(mjd), int(plate), int(fiber))

    def load_spectrum(self, index, row):
        filename = Sdss1SpectrumReader.get_filename(row['mjd'], row['plate'], row['fiber'])
        filename = os.path.join(self.path, filename)

        with fits.open(filename, memmap=False) as hdus:
            spec = self.read(hdus)
            spec.index = index
            self.read_header(hdus, row, spec)
            return spec

import os
import numpy as np
import pandas as pd

from ..spectrum import Spectrum
from .spectrumreader import SpectrumReader

class CSVSpectrumReader(SpectrumReader):
    """
    Read spectra from text files using pandas.
    """

    def __init__(self, wave_lim=None, orig=None):
        super().__init__(wave_lim=wave_lim, orig=orig)
        
        if not isinstance(orig, CSVSpectrumReader):
            pass
        else:
            pass

    def add_args(self, parser):
        super().add_args(parser)

    def init_from_args(self, args):
        super().init_from_args(args)

    def read(self, path=None):
        # TODO: extend to be able to read more columns, convert units etc
        names = ['wave', 'flux']
        df = pd.read_csv(path, comment='#', delimiter=r'\s+', names=names, )

        wave = np.array(df['wave'])
        flux = np.array(df['flux'])

        if self.wave_lim is not None:
            filt = (self.wave_lim[0] <= wave) & (wave <= self.wave_lim[1])
        else:
            filt = slice(None)

        spec = Spectrum()
        spec.wave = wave[filt]
        spec.flux = flux[filt]

        return spec

    def read_all(self, path=None):
        raise NotImplementedError()
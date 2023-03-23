import scipy.stats
import numpy as np

from .arraydataset import ArrayDataset
from ..spectrum import Spectrum

class SpectrumDataset(ArrayDataset):
    """
    Implements a dataset to store spectra with optional error and mask vectors.

    Spectra can be on the same or diffent wave grid but if wavelengths are 
    different the number of bins should still be the same as flux values are
    stored as large rectangular arrays.
    """

    PREFIX_SPECTRUMDATASET = 'spectrumdataset'

    def __init__(self, constant_wave=None, preload_arrays=False, orig=None):
        if isinstance(orig, SpectrumDataset):
            self.constant_wave = constant_wave if constant_wave is not None else orig.constant_wave
            self.wave = orig.wave
        else:
            self.constant_wave = constant_wave if constant_wave is not None else True
            self.wave = None

        super(SpectrumDataset, self).__init__(preload_arrays=preload_arrays, orig=orig)

    def init_values(self, row_count=None):
        super(SpectrumDataset, self).init_values(row_count=row_count)

        if not self.constant_wave:
            self.init_value('wave', dtype=np.float)
        self.init_value('flux', dtype=np.float)
        self.init_value('error', dtype=np.float)
        self.init_value('mask', dtype=np.int)

    def allocate_values(self, spectrum_count=None, wave_count=None):
        super(SpectrumDataset, self).allocate_values(row_count=spectrum_count)

        if spectrum_count is None:
            spectrum_count = self.row_count

        if wave_count is None:
            wave_count = self.wave.shape[-1]

        if not self.constant_wave:
            self.allocate_value('wave', shape=(wave_count,), dtype=np.float)
        self.allocate_value('flux', shape=(wave_count,), dtype=np.float)
        self.allocate_value('error', shape=(wave_count,), dtype=np.float)
        self.allocate_value('mask', shape=(wave_count,), dtype=np.int)

    def load_items(self, s=None):
        super().load_items(s)

        if self.constant_wave:
            self.wave = self.load_item('/'.join([self.PREFIX_SPECTRUMDATASET, 'wave']), np.ndarray)

    def save_items(self):
        super().save_items()

        if self.constant_wave:
            self.save_item('/'.join([self.PREFIX_SPECTRUMDATASET, 'wave']), self.wave)

    def get_value_shape(self, name=None):
        if name is None:
            return self.get_shape() + self.value_shapes['flux']
        else:
            return super().get_value_shape(name)

    def create_spectrum(self):
        # Override this function to return a specific kind of class derived from Spectrum.
        return Spectrum()

    def get_spectrum(self, i):
        spec = self.create_spectrum()

        if self.constant_wave:
            spec.wave = self.get_wave()
        else:
            spec.wave = self.get_wave(idx=i)
        spec.flux = self.get_flux(idx=i)
        spec.flux_err = self.get_error(idx=i)
        spec.mask = self.get_mask(idx=i)

        params = self.params.loc[i].to_dict()
        for p in params:
            if hasattr(spec, p):
                setattr(spec, p, params[p])

        return spec

    def get_wave(self, idx=None, chunk_size=None, chunk_id=None):
        if self.constant_wave:
            return self.wave
        else:
            return self.get_value('wave', idx, chunk_size, chunk_id)

    def set_wave(self, wave, idx=None, chunk_size=None, chunk_id=None):
        if self.constant_wave:
            self.wave = wave
            if not self.preload_arrays:
                self.save_item('/'.join([self.PREFIX_SPECTRUMDATASET, 'wave']), self.wave)
        else:
            self.set_value('wave', wave, idx, chunk_size, chunk_id)

    def get_flux(self, idx=None, chunk_size=None, chunk_id=None):
        return self.get_value('flux', idx, chunk_size, chunk_id)

    def set_flux(self, flux, idx=None, chunk_size=None, chunk_id=None):
        self.set_value('flux', flux, idx, chunk_size, chunk_id)

    def has_error(self):
        return self.has_value('error')

    def get_error(self, idx=None, chunk_size=None, chunk_id=None):
        return self.get_value('error', idx, chunk_size, chunk_id)

    def set_error(self, error, idx=None, chunk_size=None, chunk_id=None):
        self.set_value('error', error, idx, chunk_size, chunk_id)

    def has_mask(self):
        return self.has_value('mask')

    def get_mask(self, idx=None, chunk_size=None, chunk_id=None):
        return self.get_value('mask', idx, chunk_size, chunk_id)

    def set_mask(self, mask, idx=None, chunk_size=None, chunk_id=None):
        self.set_value('mask', mask, idx, chunk_size, chunk_id)

    def add_predictions(self, labels, prediction):
        # TODO: this is not used, cf: dnnmodelpredictor.py:predict
        i = 0
        for l in labels:
            if prediction.shape[2] == 1:
                self.params[l + '_pred'] = prediction[:,i,0]
            else:
                self.params[l + '_pred'] = np.mean(prediction[:,i,:], axis=-1)
                self.params[l + '_std'] = np.std(prediction[:,i,:], axis=-1)
                self.params[l + '_skew'] = scipy.stats.skew(prediction[:,i,:], axis=-1)
                self.params[l + '_median'] = np.median(prediction[:,i,:], axis=-1)

                # TODO: how to get the mode?
                # TODO: add quantiles?

            i += 1

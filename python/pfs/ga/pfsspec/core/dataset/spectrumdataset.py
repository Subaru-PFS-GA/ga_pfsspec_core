import scipy.stats
import numpy as np

from pfs.ga.pfsspec.core.util.copy import safe_deep_copy
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
        if not isinstance(orig, SpectrumDataset):
            self.constant_wave = constant_wave if constant_wave is not None else True
            self.wave = None
            self.wave_edges = None
        else:
            self.constant_wave = constant_wave if constant_wave is not None else orig.constant_wave
            self.wave = safe_deep_copy(orig.wave)
            self.wave_edges = safe_deep_copy(orig.wave_edges)

        super().__init__(preload_arrays=preload_arrays, orig=orig)

    def init_values(self, row_count=None):
        super().init_values(row_count=row_count)

        if not self.constant_wave:
            self.init_value('wave', dtype=float)
            self.init_value('wave_edges', dtype=float)
        self.init_value('flux', dtype=float)
        self.init_value('error', dtype=float)
        self.init_value('mask', dtype=int)

    def allocate_values(self, spectrum_count=None, wave_count=None):
        super().allocate_values(row_count=spectrum_count)

        if spectrum_count is None:
            spectrum_count = self.row_count

        if wave_count is None:
            wave_count = self.wave.shape[-1]

        if not self.constant_wave:
            self.allocate_value('wave', shape=(wave_count,), dtype=float)
            self.allocate_value('wave_edges', shape=(2, wave_count,), dtype=float)
        self.allocate_value('flux', shape=(wave_count,), dtype=float)
        self.allocate_value('error', shape=(wave_count,), dtype=float)
        self.allocate_value('mask', shape=(wave_count,), dtype=np.int)

    def load_items(self, s=None):
        # If we know the spectra has individual wave vectors we reinitialize the dataset value
        # dictionaries here to contain entries for wave and wave_edges
        if self.has_value('wave'):
            self.constant_wave = False
        else:
            self.constant_wave = True
        
        self.reset_values()
        self.init_values()

        super().load_items(s)

        # If there is a wave array, the wave grid is not constant
        if self.has_value('wave'):
            self.wave = None
            self.wave_edges = None
        else:
            self.wave = self.load_item('/'.join([self.PREFIX_SPECTRUMDATASET, 'wave']), np.ndarray)
            self.wave_edges = self.load_item('/'.join([self.PREFIX_SPECTRUMDATASET, 'wave_edges']), np.ndarray)

    def save_items(self):
        super().save_items()

        if self.constant_wave:
            self.save_item('/'.join([self.PREFIX_SPECTRUMDATASET, 'wave']), self.wave)
            self.save_item('/'.join([self.PREFIX_SPECTRUMDATASET, 'wave_edges']), self.wave_edges)

    def get_value_shape(self, name=None):
        if name is None:
            return self.get_shape() + self.value_shapes['flux']
        else:
            return super().get_value_shape(name)

    def create_spectrum(self):
        # Override this function to return a specific kind of class derived from Spectrum.
        return Spectrum()

    def get_spectrum(self, idx=None, chunk_size=None, chunk_id=None, trim=True):
        spec = self.create_spectrum()

        spec.append_history(f'Spectrum object of `{type(spec).__name__}` created.')

        if chunk_size is None:
            spec.id = idx
        else:
            spec.id = idx + chunk_id * chunk_size

        if self.constant_wave:
            spec.wave, spec.wave_edges, wave_mask = self.wave, self.wave_edges, None
        else:
            spec.wave, spec.wave_edges, wave_mask = self.get_wave(idx=idx, chunk_size=chunk_size, chunk_id=chunk_id)

        spec.flux = self.get_flux(idx=idx, chunk_size=chunk_size, chunk_id=chunk_id)
        spec.flux_err = self.get_error(idx=idx, chunk_size=chunk_size, chunk_id=chunk_id)
        spec.mask = self.get_mask(idx=idx, chunk_size=chunk_size, chunk_id=chunk_id)

        spec.merge_mask(wave_mask)

        # Trim the data vectors if wave contains nan values
        if wave_mask is not None and trim:
            def trim_vector(v, s):
                return None if v is None else v[s]
            s = ~np.isnan(spec.wave) & wave_mask
            spec.wave = trim_vector(spec.wave, s)
            spec.wave_edges = trim_vector(spec.wave_edges, s)
            spec.flux = trim_vector(spec.flux, s)
            spec.flux_err = trim_vector(spec.flux_err, s)
            spec.mask = trim_vector(spec.mask, s)

        params = self.get_params(None, idx=idx, chunk_size=chunk_size, chunk_id=chunk_id).to_dict()

        for p in params:
            if hasattr(spec, p):
                setattr(spec, p, params[p])

        spec.append_history(f'Wavelength range limited by dataset coverage to {spec.wave[0], spec.wave[-1]}.')
        
        if spec.wave is not None:
            spec.append_history(f'Wave vector assigned from dataset.')
        
        if spec.wave_edges is not None:
            spec.append_history(f'Wave_edges vector assigned from dataset.')

        if spec.flux is not None:
            spec.append_history(f'Flux vector loaded from dataset.')
        else:
            spec.append_history(f'No flux vector found in the dataset.')

        if spec.flux_err is not None:
            spec.append_history(f'Flux variance vector loaded from dataset.')
        else:
            spec.append_history(f'No flux variance vector found in the dataset.')

        spec.append_history(f'Spectrum loaded from dataset with ID {spec.id}, actual model parameters {spec.get_params()}.')

        return spec

    def get_wave(self, idx=None, chunk_size=None, chunk_id=None):
        if self.constant_wave:
            return self.wave, self.wave_edges, None
        else:
            wave = self.get_value('wave', idx, chunk_size, chunk_id)
            wave_edges = self.get_value('wave_edges', idx, chunk_size, chunk_id)
            mask = ~np.isnan(wave)
            return wave, wave_edges, None

    def set_wave(self, wave, wave_edges=None, idx=None, chunk_size=None, chunk_id=None):
        if self.constant_wave:
            self.wave = wave
            self.wave_edges = wave_edges
            if not self.preload_arrays:
                self.save_item('/'.join([self.PREFIX_SPECTRUMDATASET, 'wave']), self.wave)
                if wave_edges is not None:
                    self.save_item('/'.join([self.PREFIX_SPECTRUMDATASET, 'wave_edges']), self.wave)
        else:
            self.set_value('wave', wave, idx, chunk_size, chunk_id)
            if wave_edges is not None:
                if wave.ndim == wave_edges.ndim:
                    wave_edges = np.stack([wave_edges[..., :-1], wave_edges[..., 1:]], axis=-2)
                self.set_value('wave_edges', wave_edges, idx, chunk_size, chunk_id)

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

import scipy.stats
import numpy as np

from pfsspec.core import PfsObject
from .dataset import Dataset

class SpectrumDataset(PfsObject):
    """
    Implements a dataset to store spectra with optional error and mask vectors.

    Spectra can be on the same or diffent wave grid but if wavelengths are 
    different the number of bins should still be the same as flux values are
    stored as large rectangular arrays.
    """

    def __init__(self, orig=None):

        # TODO: we now have wave and three data arrays for flux, error and mask
        #       this is usually enough for training but also consider making it
        #       more generic to support continuum, etc. or entirely different types
        #       of datasets, like the implementation of grid.
        #       This could be done based on the function get_value and set_value

        if isinstance(orig, Dataset):
            self.constant_wave = orig.constant_wave
            
            self.wave = orig.wave
            self.flux = orig.flux
            self.error = orig.error
            self.mask = orig.mask
        else:
            self.constant_wave = True

            self.params = None

            self.cache_chunk_id = None
            self.cache_chunk_size = None
            self.cache_dirty = False
    
    # TODO: wrap a dataset

    # TODO: Make it more generic, support data other than spectra.
    # def init_storage(self, wcount, scount, constant_wave=True):
    #     """
    #     Initialize the storage based on the size of the data arrays. If arrays are pre-loaded,
    #     numpy ndarrays will be allocated. If the arrays are not pre-loaded and the file format is
    #     HDF5, datasets on the disk will be allocated.

    #     :param wcount: Number of wavelength bins.
    #     :param scount: Number of spectra
    #     :param constant_wave: Whether the wavelength grid is fixed.
    #     """

    #     self.constant_wave = constant_wave
    #     if self.preload_arrays:
    #         self.logger.debug('Initializing memory for dataset of size {}.'.format((scount, wcount)))

    #         if constant_wave:
    #             self.wave = np.empty(wcount)
    #         else:
    #             self.wave = np.empty((scount, wcount))
    #         self.flux = np.empty((scount, wcount))
    #         self.error = np.empty((scount, wcount))
    #         self.mask = np.empty((scount, wcount))

    #         self.logger.debug('Initialized memory for dataset of size {}.'.format((scount, wcount)))
    #     else:
    #         self.logger.debug('Allocating disk storage for dataset of size {}.'.format((scount, wcount)))

    #         if constant_wave:
    #             self.allocate_item('wave', (wcount,), np.float)
    #         else:
    #             self.allocate_item('wave', (scount, wcount), np.float)
    #         self.allocate_item('flux', (scount, wcount), np.float)
    #         self.allocate_item('error', (scount, wcount), np.float)
    #         self.allocate_item('mask', (scount, wcount), np.int32)

    #         self.logger.debug('Allocated disk storage for dataset of size {}.'.format((scount, wcount)))

    #     self.shape = (scount, wcount)

    # def load_items(self, s=None):
    #     self.params = self.load_item('params', pd.DataFrame, s=s)
    #     if self.preload_arrays or len(self.get_item_shape('wave')) == 1:
    #         # Single wave vector or preloading is requested
    #         self.wave = self.load_item('wave', np.ndarray)
    #         self.constant_wave = (len(self.wave.shape) == 1)
    #     else:
    #         self.wave = None
    #         self.constant_wave = (len(self.get_item_shape('wave')) == 1)

    #     if self.preload_arrays:
    #         self.flux = self.load_item('flux', np.ndarray, s=s)
    #         self.error = self.load_item('error', np.ndarray, s=s)
    #         if self.error is not None and np.any(np.isnan(self.error)):
    #             self.error = None
    #         self.mask = self.load_item('mask', np.ndarray, s=s)

    #         self.shape = self.flux.shape
    #     else:
    #         self.flux = None
    #         self.error = None
    #         self.mask = None

    #         self.shape = self.get_item_shape('flux')

    # def save_items(self):
    #     if self.preload_arrays:
    #         self.save_item('wave', self.wave)
    #         self.save_item('flux', self.flux)
    #         self.save_item('error', self.error)
    #         self.save_item('mask', self.error)
    #         self.save_item('params', self.params)
    #     else:
    #         if self.constant_wave:
    #             self.save_item('wave', self.wave)

    #         # Flushing the chunk cache sets the params to None so if params
    #         # is not None, it means that we have to save it now
    #         if self.params is not None:
    #             self.save_item('params', self.params)

    #         # Everything else is written to the disk lazyly

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
                self.save_item('wave', wave)
        else:
            self.set_value('wave', wave, idx, chunk_size, chunk_id)

    def get_flux(self, idx=None, chunk_size=None, chunk_id=None):
        return self.get_value('flux', idx, chunk_size, chunk_id)

    def set_flux(self, flux, idx=None, chunk_size=None, chunk_id=None):
        self.set_value('flux', flux, idx, chunk_size, chunk_id)

    def has_error(self):
        return self.has_item('error')

    def get_error(self, idx=None, chunk_size=None, chunk_id=None):
        return self.get_value('error', idx, chunk_size, chunk_id)

    def set_error(self, error, idx=None, chunk_size=None, chunk_id=None):
        self.set_value('error', error, idx, chunk_size, chunk_id)

    def has_mask(self):
        return self.has_item('mask')

    def get_mask(self, idx=None, chunk_size=None, chunk_id=None):
        return self.get_value('mask', idx, chunk_size, chunk_id)

    def set_mask(self, mask, idx=None, chunk_size=None, chunk_id=None):
        self.set_value('mask', mask, idx, chunk_size, chunk_id)

    def add_predictions(self, labels, prediction):
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

    # def reset_cache_all(self, chunk_size, chunk_id):
    #     if not self.constant_wave:
    #         self.wave = None
    #     self.flux = None
    #     self.error = None
    #     self.mask = None

    #     self.cache_chunk_id = chunk_id
    #     self.cache_chunk_size = chunk_size
    #     self.cache_dirty = False

    # def flush_cache_all(self, chunk_size, chunk_id):
    #     self.logger.debug('Flushing dataset chunk `:{}` to disk.'.format(self.cache_chunk_id))

    #     if not self.constant_wave:
    #         self.flush_cache_item('wave')
    #     self.flush_cache_item('flux')
    #     self.flush_cache_item('error')
    #     self.flush_cache_item('mask')

    #     # Sort and save the last chunk of the parameters
    #     if self.params is not None:
    #         self.params.sort_values('id', inplace=True)
    #     min_string_length = { 'interp_param': 15 }
    #     s = self.get_chunk_slice(self.cache_chunk_size, self.cache_chunk_id)
    #     self.save_item('params', self.params, s=s, min_string_length=min_string_length)

    #     # TODO: The issue with merging new chunks into an existing DataFrame is that
    #     #       the index is rebuilt repeadately, causing an even increasing processing
    #     #       time. To prevent this, we reset the params DataFrame here but it should
    #     #       only happen during building a dataset. Figure out a better solution to this.
    #     self.params = None

    #     self.logger.debug('Flushed dataset chunk {} to disk.'.format(self.cache_chunk_id))

    #     self.reset_cache_all(chunk_size, chunk_id)
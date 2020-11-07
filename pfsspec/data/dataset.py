import logging
import numpy as np
import scipy.stats
import pandas as pd

from pfsspec.pfsobject import PfsObject
from pfsspec.obsmod.spectrum import Spectrum

class Dataset(PfsObject):
    # Implements a class to store preprocessed spectra with parameters stored as
    # a pandas DataFrame. Spectra can be on the same or diffent wave grid but
    # if wavelength are different the number of bins should still be the same as
    # flux values are stored as large arrays.

    def __init__(self, preload_arrays=None, orig=None):
        super(Dataset, self).__init__(orig=orig)

        if isinstance(orig, Dataset):
            self.preload_arrays = preload_arrays or orig.preload_arrays
            self.constant_wave = orig.constant_wave
            self.shape = orig.shape

            self.params = orig.params.copy()
            self.wave = orig.wave
            self.flux = orig.flux
            self.error = orig.error
            self.mask = orig.mask

            self.cache_chunk_id = orig.cache_chunk_id
            self.cache_chunk_size = orig.cache_chunk_size
        else:
            self.preload_arrays = preload_arrays or False
            self.constant_wave = True
            self.shape = None

            self.params = None
            self.wave = None
            self.flux = None
            self.error = None
            self.mask = None

            self.cache_chunk_id = None
            self.cache_chunk_size = None
            
    #region Load and save

    def init_storage(self, wcount, scount, constant_wave=True):
        self.constant_wave = constant_wave
        if self.preload_arrays:
            logging.debug('Initializing memory for dataset of size {}.'.format((scount, wcount)))

            if constant_wave:
                self.wave = np.empty(wcount)
            else:
                self.wave = np.empty((scount, wcount))
            self.flux = np.empty((scount, wcount))
            self.error = np.empty((scount, wcount))
            self.mask = np.empty((scount, wcount))

            logging.debug('Initialized memory for dataset of size {}.'.format((scount, wcount)))
        else:
            logging.debug('Allocating disk storage for dataset of size {}.'.format((scount, wcount)))

            if constant_wave:
                self.allocate_item('wave', (wcount,), np.float)
            else:
                self.allocate_item('wave', (scount, wcount), np.float)
            self.allocate_item('flux', (scount, wcount), np.float)
            self.allocate_item('error', (scount, wcount), np.float)
            self.allocate_item('mask', (scount, wcount), np.int32)

            logging.debug('Allocated disk storage for dataset of size {}.'.format((scount, wcount)))

        self.shape = (scount, wcount)

    def load(self, filename, format='h5', s=None):
        super(Dataset, self).load(filename, format=format, s=s)

        logging.info("Loaded dataset with shapes:")
        logging.info("  params:  {}".format(self.params.shape))
        logging.info("  columns: {}".format(self.params.columns))

    def load_items(self, s=None):
        self.params = self.load_item('params', pd.DataFrame, s=s)
        if self.preload_arrays or len(self.get_item_shape('wave')) == 1:
            # Single wave vector or preloading is requested
            self.wave = self.load_item('wave', np.ndarray)
            self.constant_wave = (len(self.wave.shape) == 1)
        else:
            self.wave = None
            self.constant_wave = (len(self.get_item_shape('wave')) == 1)

        if self.preload_arrays:
            self.flux = self.load_item('flux', np.ndarray, s=s)
            self.error = self.load_item('error', np.ndarray, s=s)
            if self.error is not None and np.any(np.isnan(self.error)):
                self.error = None
            self.mask = self.load_item('mask', np.ndarray, s=s)

            self.shape = self.flux.shape
        else:
            self.flux = None
            self.error = None
            self.mask = None

            self.shape = self.get_item_shape('flux')

    def save(self, filename, format='h5'):
        super(Dataset, self).save(filename, format=format)

        logging.info("Saved dataset with shapes:")
        logging.info("  params:  {}".format(self.params.shape))
        logging.info("  columns: {}".format(self.params.columns))

    def save_items(self):
        self.save_item('params', self.params)
        
        if self.preload_arrays:
            self.save_item('wave', self.wave)
            self.save_item('flux', self.flux)
            self.save_item('error', self.error)
            self.save_item('mask', self.error)
        else:
            if self.constant_wave:
                self.save_item('wave', self.wave)
            else:
                pass

            # Everything else is written to the disk lazyly

    #endregion
    #region Params access

    def get_params(self, labels, idx=None, chunk_size=None, chunk_id=None):
        if chunk_id is None:
            return self.params[labels].iloc[idx]
        else:
            return self.params[labels].iloc[chunk_id * chunk_size + idx]

    def set_params(self, labels, values, idx=None, chunk_size=None, chunk_id=None):
        for i, label in enumerate(labels):
            if label not in self.params.columns:
                self.params.loc[:, label] = np.zeros((), dtype=values.dtype)
            
            if chunk_id is None:
                self.params[label].iloc[idx] = values[..., i]
            else:
                self.params[label].iloc[chunk_id * chunk_size + idx] = values[..., i]

    #endregion
    #region Array access

    def get_chunk_slice(self, chunk_size, chunk_id):
        if chunk_id is None:
            return np.s_[()]
        else:
            fr = chunk_id * chunk_size
            to = min((chunk_id + 1) * chunk_size, self.shape[0])
            return np.s_[fr:to]

    def check_cache(self, name, chunk_size, chunk_id):
        # Flush cache if necessary
        data = getattr(self, name)

        if self.cache_chunk_id != chunk_id or self.current_chunk_size != chunk_size:
            data = None

            if not self.constant_wave:
                self.wave = None
            self.flux = None
            self.error = None
            self.mask = None

            self.current_chunk_id = chunk_id
            self.current_chunk_size = chunk_size
        
        # Load and cache current
        if data is None:
            s = self.get_chunk_slice(chunk_size, chunk_id)
            data = self.load_item(name, np.ndarray, s=s)
        
        setattr(self, name, data)

    def get_item(self, name, idx=None, chunk_size=None, chunk_id=None):
        if not self.preload_arrays:
            if chunk_size is None:
                # No chunking, load directly from storage
                # Assume idx is absolute within file and not relative to chunk
                data = self.load_item(name, np.ndarray, s=idx)
                return data
            else:
                # Chunked lazy loading, use cache
                # Assume idx is relative to chunk
                self.check_cache(name, chunk_size, chunk_id)
        # Return slice from cache
        return getattr(self, name)[idx if idx is not None else ()]

    def set_item(self, name, data, idx=None, chunk_size=None, chunk_id=None):
        if self.preload_arrays:
            getattr(self, name)[idx if idx is not None else ()] = data
        elif chunk_size is not None:
            # Load necessary chunk, update and save
            # Assume index is relative to chunk
            s = self.get_chunk_slice(chunk_size, chunk_id)
            v = self.load_item(name, np.ndarray, s=s)
            v[idx] = data
            self.save_item(name, v, s=s)
        else:
            # No chunking, write directly to storage
            # Assume idx is absolute to file
            self.save_item(name, data, s=idx)

    def has_item(self, name):
        if self.preload_arrays:
            return getattr(self, name) is not None
        else:
            return super(Dataset, self).has_item(name)

    def get_wave(self, idx=None, chunk_size=None, chunk_id=None):
        if self.constant_wave:
            return self.wave
        else:
            return self.get_item('wave', idx, chunk_size, chunk_id)

    def set_wave(self, wave, idx=None, chunk_size=None, chunk_id=None):
        if self.constant_wave:
            self.wave = wave
        else:
            self.set_item('wave', wave, idx, chunk_size, chunk_id)

    def get_flux(self, idx=None, chunk_size=None, chunk_id=None):
        return self.get_item('flux', idx, chunk_size, chunk_id)

    def set_flux(self, flux, idx=None, chunk_size=None, chunk_id=None):
        self.set_item('flux', flux, idx, chunk_size, chunk_id)

    def has_error(self):
        return self.has_item('error')

    def get_error(self, idx=None, chunk_size=None, chunk_id=None):
        return self.get_item('error', idx, chunk_size, chunk_id)

    def set_error(self, error, idx=None, chunk_size=None, chunk_id=None):
        self.set_item('error', error, idx, chunk_size, chunk_id)

    def has_mask(self):
        return self.has_item('mask')

    def get_mask(self, idx=None, chunk_size=None, chunk_id=None):
        return self.get_item('mask', idx, chunk_size, chunk_id)

    def set_mask(self, mask, idx=None, chunk_size=None, chunk_id=None):
        self.set_item('mask', mask, idx, chunk_size, chunk_id)

    #endregion

    def get_count(self):
        return self.params.shape[0]

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

    ###
    # TODO: rewrite these to work with la

    def reset_index(self, df):
        df.index = pd.RangeIndex(len(df.index))

    def get_split_index(self, split_value):
        split_index = int((1 - split_value) *  self.get_count())
        return split_index

    def get_split_ranges(self, split_index):
        a_range = pd.Series(split_index * [True] + (self.get_count() - split_index) * [False])
        b_range = ~a_range
        return a_range, b_range

    def split(self, split_value):
        split_index = self.get_split_index(split_value)
        a_range, b_range = self.get_split_ranges(split_index)

        a = self.where(a_range)
        b = self.where(b_range)

        return split_index, a, b

    def where(self, f):
        if f is None:
            ds = type(self)(orig=self)
            return ds
        else:
            ds = type(self)(orig=self)
            ds.params = self.params[f]
            self.reset_index(ds.params)

            if self.preload_arrays:
                if self.constant_wave:
                    ds.wave = self.wave
                else:
                    ds.wave = self.wave[f]
                ds.flux = self.flux[f]
                ds.error = self.error[f] if self.error is not None else None
                ds.mask = self.mask[f] if self.mask is not None else None
            else:
                if self.constant_wave:
                    ds.wave = self.wave
                else:
                    ds.wave = None
                ds.flux = None
                ds.error = None
                ds.mask = None

            ds.shape = (ds.params.shape[0],) + self.shape[1:]
            
            return ds

    def merge(self, b):
        ds = Dataset()

        ds.params = pd.concat([self.params, b.params], axis=0)
        self.reset_index(ds.params)

        if self.preload_arrays:
            if self.constant_wave:
                ds.wave = self.wave
            else:
                ds.wave = np.concatenate([self.wave, b.wave], axis=0)
            ds.flux = np.concatenate([self.flux, b.flux], axis=0)
            ds.error = np.concatenate([self.error, b.error], axis=0) if self.error is not None and b.error is not None else None
            ds.mask = np.concatenate([self.mask, b.mask], axis=0) if self.mask is not None and b.mask is not None else None
        else:
            if self.constant_wave:
                ds.wave = self.wave
            else:
                ds.wave = None
            ds.flux = None
            ds.error = None
            ds.mask = None

        ds.shape = (ds.params.shape[0],) + self.shape[1:]

        return ds

    ###

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


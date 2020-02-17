import logging
import numpy as np
import scipy.stats
import pandas as pd

from pfsspec.pfsobject import PfsObject
from pfsspec.obsmod.spectrum import Spectrum

class Dataset(PfsObject):
    def __init__(self, orig=None):
        if orig is None:
            self.params = None
            self.wave = None
            self.flux = None
            self.error = None
            self.mask = None
            self.U = None
            self.S = None
            self.V = None
            self.PC = None
        else:
            self.params = orig.params.copy()
            self.wave = orig.wave
            self.flux = orig.flux
            self.error = orig.error
            self.mask = orig.mask
            self.U = None
            self.S = None
            self.V = None
            self.PC = None

    def get_spectrum_count(self):
        if filter is not None:
            return self.params[filter].shape[0]
        else:
            return self.params.shape[0]

    def create_spectrum(self):
        return Spectrum()

    def get_spectrum(self, i):
        spec = self.create_spectrum()

        if self.wave.ndim == 1:
            spec.wave = self.wave
        else:
            spec.wave = self.wave[i]
        spec.flux = self.flux[i]
        if self.error is not None:
            spec.flux_err = self.error[i]
        if self.mask is not None:
            spec.mask = self.mask[i]

        params = self.params.loc[i].to_dict()
        for p in params:
            if hasattr(spec, p):
                setattr(spec, p, params[p])

        return spec

    def init_storage(self, wcount, scount, constant_wave=True):
        logging.debug('Initializing memory for dataset of size {}.'.format((scount, wcount)))

        if constant_wave:
            self.wave = np.empty(wcount)
        else:
            self.wave = np.empty((scount, wcount))
        self.flux = np.empty((scount, wcount))
        self.error = np.empty((scount, wcount))
        self.mask = np.empty((scount, wcount))

        logging.debug('Initialized memory for dataset of size {}.'.format((scount, wcount)))

    def save_items(self):
        self.save_item('params', self.params)
        self.save_item('wave', self.wave)
        self.save_item('flux', self.flux)
        self.save_item('error', self.error)
        self.save_item('mask', self.error)

    def load(self, filename, format='pickle'):
        super(Dataset, self).load(filename, format)

        logging.info("Loaded dataset with shapes:")
        logging.info("  params:  {}".format(self.params.shape))
        logging.info("  wave:    {}".format(self.wave.shape))
        logging.info("  flux:    {}".format(self.flux.shape))
        logging.info("  error:   {}".format(self.error.shape if self.error is not None else "None"))
        logging.info("  mask:    {}".format(self.mask.shape if self.mask is not None else "None"))
        logging.info("  columns: {}".format(self.params.columns))

    def load_items(self, s=None):
        self.params = self.load_item('params', pd.DataFrame)
        self.wave = self.load_item('wave', np.ndarray)
        self.flux = self.load_item('flux', np.ndarray)
        self.error = self.load_item('error', np.ndarray)
        if self.error is not None and np.any(np.isnan(self.error)):
            self.error = None
        self.mask = self.load_item('mask', np.ndarray)

    def reset_index(self, df):
        df.index = pd.RangeIndex(len(df.index))

    def get_split_index(self, split_value):
        split_index = int((1 - split_value) *  self.flux.shape[0])
        return split_index

    def get_split_ranges(self, split_index):
        a_range = [i for i in range(0, split_index)]
        b_range = [i for i in range(split_index, self.flux.shape[0])]
        return a_range, b_range

    def split(self, split_value):
        split_index = self.get_split_index(split_value)
        a_range, b_range = self.get_split_ranges(split_index)

        a = self.filter(a_range)
        b = self.filter(b_range)

        return split_index, a, b

    def filter(self, f):
        if f is None:
            return self
        else:
            ds = Dataset()
            ds.params = self.params.iloc[f]
            self.reset_index(ds.params)
            if self.wave.ndim == 1:
                ds.wave = self.wave
            else:
                ds.wave = self.wave[f]
            ds.flux = self.flux[f]
            ds.error = self.error[f] if self.error is not None else None
            ds.mask = self.mask[f] if self.mask is not None else None
            return ds

    def merge(self, b):
        ds = Dataset()

        ds.params = pd.concat([self.params, b.params], axis=0)
        self.reset_index(ds.params)
        if self.wave.ndim == 1:
            ds.wave = self.wave
        else:
            ds.wave = np.concatenate([self.wave, b.wave], axis=0)
        ds.flux = np.concatenate([self.flux, b.flux], axis=0)
        ds.error = np.concatenate([self.error, b.error], axis=0) if self.error is not None and b.error is not None else None
        ds.mask = np.concatenate([self.mask, b.mask], axis=0) if self.mask is not None and b.mask is not None else None

        return ds

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

    def run_pca(self, truncate=None):
        if self.wave.ndim != 1:
            raise Exception('PCA is only meaningful with a constant wave grid.')

        C = np.dot(self.flux.transpose(), self.flux)
        self.U, self.S, self.V = np.linalg.svd(C)

        if truncate is not None:
            self.PC = np.dot(self.flux, self.U[:, 0:truncate])
        else:
            self.PC = np.dot(self.flux, self.U[:, :])

        self.wave = np.arange(self.PC.shape[1])
        self.flux = self.PC
        self.error = np.zeros(self.flux.shape)
        self.mask = np.full(self.flux.shape, False)

    def save_pca(self, filename, format=None):
        logging.info("Saving PCA eigensystem to file {}...".format(filename))
        self.save(filename, format=format, save_items_func=self.save_pca_items)
        logging.info("Saved PCA eigensystem.")

    def save_pca_items(self):
        self.save_item('U', self.U)
        self.save_item('S', self.S)
        self.save_item('V', self.V)

    def load_pca(self, filename, s=None, format=None):
        logging.info("Loading PCA eigensystem from file {}...".format(filename))
        self.load(filename, s=s, format=format, load_items_func=self.load_pca_items)
        logging.info("Loaded PCA eigensystem.")

    def load_pca_items(self, s=None):
        self.U = self.load_item('U', np.ndarray)
        self.S = self.load_item('S', np.ndarray)
        self.V = self.load_item('V', np.ndarray)


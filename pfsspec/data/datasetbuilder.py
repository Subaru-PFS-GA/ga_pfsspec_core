import sys
import logging
import numpy as np
import pandas as pd

from pfsspec.parallel import srl_map, prll_map
from pfsspec.data.dataset import Dataset

class DatasetBuilder():
    def __init__(self, orig=None, random_seed=None):
        if orig is not None:
            self.random_seed = random_seed or orig.random_seed
            self.random_state = None
            self.dataset = orig.dataset
            self.parallel = orig.parallel
            self.params = orig.params
            self.pipeline = orig.pipeline
        else:
            self.random_seed = random_seed
            self.dataset = None
            self.parallel = True
            self.params = None
            self.pipeline = None

    def add_args(self, parser):
        pass

    def init_from_args(self, args):
        # Only allow parallel if random seed is not set
        # It would be very painful to reproduce the same dataset with multiprocessing
        self.parallel = ('debug' not in args or not args['debug']) and self.random_seed is None
        if not self.parallel:
            logging.info('Dataset builder running in sequential mode.')

    def get_spectrum_count(self):
        raise NotImplementedError()

    def get_wave_count(self):
        raise NotImplementedError()

    def create_dataset(self, init_storage=True):
        dataset = Dataset()
        if init_storage:
            dataset.init_storage(self.get_wave_count(), self.get_spectrum_count())
            # This is used with surveys
            logging.info('Creating dataset with shapes {} {}.'.format(dataset.wave.shape, dataset.flux.shape))
        return dataset

    def init_process(self, i):
        if self.random_seed is not None:
            self.random_state = np.random.RandomState(self.random_seed + i + 1)
        else:
            self.random_state = np.random.RandomState(None)

    def process_item(self, i):
        raise NotImplementedError()

    def build(self):
        self.random_state = np.random.RandomState(self.random_seed)
        self.dataset = self.create_dataset()

        if self.parallel:
            spectra = prll_map(self.init_process, self.process_item, range(self.get_spectrum_count()), verbose=True)
        else:
            spectra = srl_map(self.init_process, self.process_item, range(self.get_spectrum_count()), verbose=True)

        # Sort spectra by id
        # Here we assume that params is also sorted on the id column (not the default index!)
        if self.params is not None and spectra[0].id is not None:
            spectra.sort(key=lambda s: s.id)

        for i in range(len(spectra)):
            self.dataset.flux[i, :] = spectra[i].flux
            self.dataset.error[i, :] = spectra[i].flux_err

        return spectra

    def copy_params_from_spectra(self, spectra):
        columns = spectra[0].get_param_names()
        names = []
        data = []
        for p in columns:
            v = getattr(spectra[0], p)
            # If parameter is an array, assume it's equal length and copy to
            # pandas as a set of columns instead of a single column
            if v is None:
                names.append(p)
                data.append([np.nan for s in spectra])
            elif isinstance(v, np.ndarray):
                if len(v.shape) > 1:
                    raise Exception('Can only serialize one-dimensional arrays')
                for i in range(v.shape[0]):
                    names.append('{}_{}'.format(p, i))
                    data.append([getattr(s, p)[i] for s in spectra])
            else:
                names.append(p)
                data.append([getattr(s, p) for s in spectra])
        self.dataset.params = pd.DataFrame(list(zip(*data)), columns=names)
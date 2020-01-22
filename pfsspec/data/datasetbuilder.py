import sys
import os
import logging
import numpy as np
import pandas as pd

from pfsspec.parallel import SmartParallel, prll_map
from pfsspec.data.dataset import Dataset
import pfsspec.util as util

class DatasetBuilder():
    def __init__(self, orig=None, random_seed=None):
        if orig is not None:
            self.random_seed = random_seed or orig.random_seed
            self.random_state = None
            self.dataset = orig.dataset
            self.parallel = orig.parallel
            self.match_params = orig.params
            self.pipeline = orig.pipeline
        else:
            self.random_seed = random_seed
            self.random_state = None
            self.dataset = None
            self.parallel = True
            self.match_params = None
            self.pipeline = None

    def get_arg(self, name, old_value, args=None):
        args = args or self.args
        return util.get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        args = args or self.args
        return util.is_arg(name, args)

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
            constant_wave = self.pipeline.is_constant_wave()
            dataset.init_storage(self.get_wave_count(), self.get_spectrum_count(), constant_wave=constant_wave)
        return dataset

    def init_process(self):
        # NOTE: cannot initialize class-specific data here because the initializer function is executed inside
        #       the worker process before the class data is copied over (although not sure why, since the
        #       process is supposed to be forked rather than a new one started...)
        pass

    def init_random_state(self):
        if self.random_state is None:
            if self.random_seed is not None:
                # NOTE: this seed won't be consistent across runs because pids can vary
                self.random_state = np.random.RandomState(self.random_seed + os.getpid() + 1)
            else:
                self.random_state = np.random.RandomState(None)
            self.pipeline.random_state = self.random_state
            logging.debug("Initialized random state on pid {}, rnd={}".format(os.getpid(), self.random_state.rand()))

    def process_item(self, i):
        self.init_random_state()

    def do_nothing(self, i):
        print(i)

    def build(self):
        self.dataset = self.create_dataset()

        logging.info('Building dataset of size {}'.format(self.dataset.flux.shape))

        rng = range(self.get_spectrum_count())
        with SmartParallel(initializer=self.init_process, verbose=True, parallel=self.parallel) as p:
            spectra = [ s for s in p.map(self.process_item, rng) ]

        # Sort spectra by id
        # Here we assume that params is also sorted on the id column (not the default index!)
        if self.match_params is not None and spectra[0].id is not None:
            spectra.sort(key=lambda s: s.id)

        for i in range(len(spectra)):
            if self.dataset.wave.ndim != 1:
                self.dataset.wave[i, :] = spectra[i].wave
            else:
                self.dataset.wave[:] = self.pipeline.get_wave()
            self.dataset.flux[i, :] = spectra[i].flux
            self.dataset.error[i, :] = spectra[i].flux_err

        self.copy_params_from_spectra(spectra)

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
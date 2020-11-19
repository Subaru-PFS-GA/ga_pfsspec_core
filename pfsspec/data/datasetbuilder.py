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
            self.parallel = orig.parallel
            self.threads = orig.threads
            self.resume = orig.resume
            self.match_params = orig.params
            self.pipeline = orig.pipeline
            self.dataset = orig.dataset if dataset is None else dataset
        else:
            self.random_seed = random_seed
            self.random_state = None
            self.parallel = True
            self.threads = None
            self.resume = False
            self.match_params = None
            self.pipeline = None
            self.dataset = None

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
        self.threads = self.get_arg('threads', self.threads, args)
        self.resume = self.get_arg('resume', self.resume, args)

    def create_dataset(self, preload_arrays=False):
        return Dataset(preload_arrays=preload_arrays)

    def get_spectrum_count(self):
        raise NotImplementedError()

    def get_wave_count(self):
        raise NotImplementedError()

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

    def store_item(self, i, spec):
        s = np.s_[i, :]

        # TODO: when implementing continue, first item test should be different
        if i == 0 and self.dataset.constant_wave:
            self.dataset.set_wave(spec.wave)
            # self.pipeline.get_wave()
        elif not self.dataset.constant_wave:
            self.dataset.set_wave(spec.wave, idx=s)

        self.dataset.set_flux(spec.flux, idx=s)
        if spec.flux_err is not None:
            self.dataset.set_error(spec.flux_err, idx=s)
        if spec.mask is not None:
            self.dataset.set_mask(spec.mask, idx=s)

        row = spec.get_params_as_datarow()
        self.dataset.set_params_row(row)

    def process_and_store_item(self, i):
        spec = self.process_item(i)
        self.store_item(i, spec)
        return spec

    def do_nothing(self, i):
        print(i)

    def build(self):
        if not self.resume:
            logging.info('Building a new dataset of size {}'.format(self.dataset.shape))
            rng = range(self.get_spectrum_count())
        else:
            logging.info('Resume building a dataset of size {}'.format(self.dataset.shape))
            all = set(range(self.get_spectrum_count()))
            existing = set(self.dataset.params['id'])
            rng = list(all - existing)
                
        with SmartParallel(initializer=self.init_process, verbose=True, parallel=self.parallel, threads=self.threads) as p:
            for s in p.map(self.process_item, rng):
                self.store_item(s.id, s)

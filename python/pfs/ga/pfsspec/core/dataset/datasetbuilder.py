import sys
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

import pfs.ga.pfsspec.core.util as util
from pfs.ga.pfsspec.core.util.smartparallel import SmartParallel
from pfs.ga.pfsspec.core import PfsObject
from .dataset import Dataset

class DatasetBuilder(PfsObject):
    def __init__(self, orig=None, random_seed=None):
        super(DatasetBuilder, self).__init__(orig=orig)

        if isinstance(orig, PfsObject):
            self.random_seed = random_seed or orig.random_seed
            self.random_state = None
            self.parallel = orig.parallel
            self.threads = orig.threads
            self.resume = orig.resume
            self.chunk_size = orig.chunk_size
            self.top = orig.top
            self.match_params = orig.match_params
            self.pipeline = orig.pipeline
            self.dataset = orig.dataset
        else:
            self.random_seed = random_seed
            self.random_state = None
            self.parallel = True
            self.threads = None
            self.resume = False
            self.chunk_size = None
            self.top = None
            self.match_params = None
            self.pipeline = None
            self.dataset = None

    def get_arg(self, name, old_value, args=None):
        args = args or self.args
        return util.args.get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        args = args or self.args
        return util.args.is_arg(name, args)

    def add_args(self, parser):
        parser.add_argument('--chunk-size', type=int, help='Dataset chunk size.\n')
        parser.add_argument('--top', type=int, help='Stop after this many items.\n')

    def init_from_args(self, config, args):
        # Only allow parallel if random seed is not set
        # It would be very painful to reproduce the same dataset with multiprocessing
        self.threads = self.get_arg('threads', self.threads, args)
        self.parallel = self.random_seed is None and (self.threads is None or self.threads > 1)
        if not self.parallel:
            self.logger.info('Dataset builder running in sequential mode.')
        self.resume = self.get_arg('resume', self.resume, args)
        self.chunk_size = self.get_arg('chunk_size', self.chunk_size, args)
        if self.chunk_size == 0:
            self.chunk_size = None
        self.top = self.get_arg('top', self.top, args)

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
            self.logger.debug("Initialized random state on pid {}, rnd={}".format(os.getpid(), self.random_state.rand()))

    def process_item(self, i):
        self.init_random_state()

    def store_item(self, i, spec):
        if self.chunk_size is None:
            chunk_size = None
            chunk_id = None
            s = np.s_[i, :]
        else:
            # When chunking, use chunk-relative index
            chunk_size = self.chunk_size
            chunk_id, ii = self.dataset.get_chunk_id(i, self.chunk_size)
            s = np.s_[ii, :]

        # TODO: when implementing continue, first item test should be different
        if i == 0 and self.dataset.constant_wave:
            self.dataset.set_wave(spec.wave)
            # self.pipeline.get_wave()
        elif not self.dataset.constant_wave:
            self.dataset.set_wave(spec.wave, idx=s, chunk_size=chunk_size, chunk_id=chunk_id)

        self.dataset.set_flux(spec.flux, idx=s, chunk_size=chunk_size, chunk_id=chunk_id)
        if spec.flux_err is not None:
            self.dataset.set_error(spec.flux_err, idx=s, chunk_size=chunk_size, chunk_id=chunk_id)
        if spec.mask is not None:
            self.dataset.set_mask(spec.mask, idx=s, chunk_size=chunk_size, chunk_id=chunk_id)

        # TODO: chunking here
        row = spec.get_params_as_datarow()
        self.dataset.append_params_row(row)

    def process_and_store_item(self, i):
        spec = self.process_item(i)
        self.store_item(i, spec)
        return spec

    def build(self):
        # If chunking is used, make sure that spectra are processed in batches so no frequent
        # cache misses occur at the boundaries of the chunks.
        
        count = self.get_spectrum_count()
        chunk_ranges = []
        total_items = 0

        if self.chunk_size is not None:
            # TODO: this is a restriction when preparing data sets based on
            #       surveys where spectrum counts will never match chunking
            if count % self.chunk_size != 0:
                raise Exception('Total count must be a multiple of chunk size.')

            for chunk_id in range(count // self.chunk_size):
                rng = range(chunk_id * self.chunk_size, (chunk_id + 1) * self.chunk_size)
                chunk_ranges.append(rng)
                total_items += len(rng)
        else:
            rng = range(count)
            chunk_ranges.append(rng)
            total_items += len(rng)

        if not self.resume:
            self.logger.info('Building a new dataset of size {}'.format(self.dataset.get_value_shape('flux')))
        else:
            self.logger.info('Resume building a dataset of size {}'.format(self.dataset.get_value_shape('flux')))
            existing = set(self.dataset.params['id'])
            total_items = 0
            for chunk_id in range(count // self.chunk_size):
                all = set(chunk_ranges[chunk_id])
                rng = list(all - existing)
                chunk_ranges[chunk_id] = rng
                total_items += len(rng)

        t = tqdm(total=total_items)
        if self.chunk_size is not None:
            # Run processing in chunks, data is written to the disk continuously
            for rng in chunk_ranges:
                if len(rng) > 0:
                    with SmartParallel(initializer=self.init_process, verbose=False, parallel=self.parallel, threads=self.threads) as p:
                        for s in p.map(self.process_item, rng):
                            self.store_item(s.id, s)
                            t.update(1)
                    self.dataset.flush_cache_all(None, None)
        else:
            # Run in a single chunk, everything will be saved at the end
            with SmartParallel(initializer=self.init_process, verbose=False, parallel=self.parallel, threads=self.threads) as p:
                    for s in p.map(self.process_item, chunk_ranges[0]):
                        self.store_item(s.id, s)
                        t.update(1)

        # Sort dataset parameters which can be shuffled due to parallel execution
        # If we write params to the HDF5 file directly, params will be None so
        # we don't need to sort.
        if self.dataset.params is not None:
            self.dataset.params.sort_index(inplace=True)

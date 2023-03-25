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

        if not isinstance(orig, PfsObject):
            self.random_seed = random_seed
            self.random_state = None
            self.parallel = True
            self.threads = None
            self.resume = False
            self.chunk_size = None
            self.top = None
            self.labels = None

            self.dataset = None
        else:
            self.random_seed = random_seed or orig.random_seed
            self.random_state = None
            self.parallel = orig.parallel
            self.threads = orig.threads
            self.resume = orig.resume
            self.chunk_size = orig.chunk_size
            self.top = orig.top
            self.labels = orig.labels
            
            self.dataset = orig.dataset

    def get_arg(self, name, old_value, args=None):
        args = args or self.args
        return util.args.get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        args = args or self.args
        return util.args.is_arg(name, args)

    def add_args(self, parser):
        parser.add_argument('--chunk-size', type=int, help='Dataset chunk size.\n')
        parser.add_argument('--top', type=int, help='Stop after this many items.\n')

        # A generic parameter to create extra dataset columns (labels) with python functions.
        parser.add_argument('--label', action='append', nargs=2, metavar=('name', 'function'), help='Additional dataset label from python expression.\n')

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

        # Additional labels
        self.labels = self.get_arg('label', self.labels, args)

    def create_dataset(self, preload_arrays=False):
        return Dataset(preload_arrays=preload_arrays)

    # TODO: spectrum specific, move to subclass
    def get_spectrum_count(self):
        raise NotImplementedError()

    # TODO: spectrum specific, move to subclass
    def get_wave_count(self):
        raise NotImplementedError()

    def init_process(self, worker_id):
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
            
            self.logger.debug("Initialized random state on pid {}, rnd={}".format(os.getpid(), self.random_state.rand()))

    def process_item(self, i):
        self.init_random_state()

    def store_item(self, i, spec):
        # TODO: make this a function callable from derived classes
        if self.chunk_size is None:
            chunk_size = None
            chunk_id = None
            s = np.s_[i, :]
        else:
            # When chunking, use chunk-relative index
            chunk_size = self.chunk_size
            chunk_id, ii = self.dataset.get_chunk_id(i, self.chunk_size)
            s = np.s_[ii, :]

        # TODO: much of it needs to be moved to SpectrumDatasetBuilder because it's
        #       spectrum specific
        # TODO: when implementing continue, first item test should be different
        if i == 0 and self.dataset.constant_wave:
            self.dataset.set_wave(spec.wave)
        elif not self.dataset.constant_wave:
            self.dataset.set_wave(spec.wave, idx=s, chunk_size=chunk_size, chunk_id=chunk_id)

        self.dataset.set_flux(spec.flux, idx=s, chunk_size=chunk_size, chunk_id=chunk_id)
        if spec.flux_err is not None:
            self.dataset.set_error(spec.flux_err, idx=s, chunk_size=chunk_size, chunk_id=chunk_id)
        if spec.mask is not None:
            self.dataset.set_mask(spec.mask, idx=s, chunk_size=chunk_size, chunk_id=chunk_id)

        # TODO: chunking here?
        row = spec.get_params_as_datarow()

        # TODO: move append params logic from dataset here.
        self.dataset.append_params_row(row)

    def process_and_store_item(self, i):
        spec = self.process_item(i)
        self.store_item(i, spec)
        return spec

    def build(self):
        # If chunking is used, make sure that spectra are processed in batches so no frequent
        # cache misses occur at the boundaries of the chunks.
        
        total_items, chunk_ranges = self.determine_chunk_ranges()
        self.process_items(total_items, chunk_ranges)
        if self.labels is not None:
            self.process_labels()

    def determine_chunk_ranges(self):
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
            # TODO: "flux" is spectrum specific, consider generalizing
            self.logger.info('Building a new dataset of size {}'.format(self.dataset.get_value_shape('flux')))
        else:
            # TODO: "flux" is spectrum specific, consider generalizing
            self.logger.info('Resume building a dataset of size {}'.format(self.dataset.get_value_shape('flux')))
            existing = set(self.dataset.params['id'])
            total_items = 0
            for chunk_id in range(count // self.chunk_size):
                all = set(chunk_ranges[chunk_id])
                rng = list(all - existing)
                chunk_ranges[chunk_id] = rng
                total_items += len(rng)
        
        return total_items, chunk_ranges

    def process_items(self, total_items, chunk_ranges):
        # Call the processing function for each items, either sequentially or in parallel

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

        # TODO: cf comments in Dataset.append_params_row

        if isinstance(self.dataset.params, dict):
            self.dataset.params = pd.DataFrame(self.dataset.params, index=self.dataset.params['id'])
            self.dataset.params.sort_index(inplace=True)

    def process_labels(self):
        # Execute the additional column generator expression provided on the command-line

        # Apply each expression on the 
        for [name, exp] in self.labels:
            # Take a the list of columns, these will be the parameters to the lambda we
            # convert the provided expression into. The expression cannot be reused
            # between calls within this loop because the list of columns changes over
            # iterations.

            # TODO: here we assume the column names are proper variable names,
            #       escape column names if necessary.
            #       also avoid using python reserved words, like class etc.
            
            src = "lambda {}: {}".format(', '.join([c for c in self.dataset.params.columns]), exp)
            fun = eval(src)
            self.dataset.params[name] = self.dataset.params.apply(lambda r: fun(**r.to_dict()), axis=1)
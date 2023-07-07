import sys
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from pfs.ga.pfsspec.core.scripts import Plugin
from pfs.ga.pfsspec.core.util.smartparallel import SmartParallel
from .dataset import Dataset

class DatasetBuilder(Plugin):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, DatasetBuilder):
            self.chunk_size = None
            self.labels = None

            self.dataset = None
        else:
            self.chunk_size = orig.chunk_size
            self.labels = orig.labels
            
            self.dataset = orig.dataset

    def add_args(self, parser, config):
        super().add_args(parser, config)

        parser.add_argument('--chunk-size', type=int, help='Dataset chunk size.\n')

        # A generic parameter to create extra dataset columns (labels) with python functions.
        parser.add_argument('--label', action='append', nargs=2, metavar=('name', 'function'), help='Additional dataset label from python expression.\n')

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)
        
        self.chunk_size = self.get_arg('chunk_size', self.chunk_size, args)
        if self.chunk_size == 0:
            self.chunk_size = None

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

    def process_item(self, i):
        raise NotImplementedError()
    
    def process_item_error(self, ex, i):
        raise NotImplementedError()

    def store_item(self, i, spec):
        raise NotImplementedError()

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
                        for s in p.map(self.process_item, self.process_item_error, rng):
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

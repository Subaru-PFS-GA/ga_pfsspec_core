import os
import logging
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import pfsspec.core.util as util
from .dataset import Dataset

class DatasetMerger():
    def __init__(self, orig=None):
        if isinstance(orig, DatasetMerger):
            self.input_paths = orig.input_paths
            self.input_datasets = orig.input_datasets
            self.input_constant_wave = orig.input_constant_wave
            self.input_wave_count = orig.input_wave_count
            self.input_data_count = orig.input_data_count
            self.input_chunk_count = orig.input_chunk_count
            self.output_path = orig.output_path
            self.output_dataset = orig.output_dataset
            self.preload_arrays = orig.preload_arrays
            self.chunk_size = orig.chunk_size
        else:
            self.input_paths = None
            self.input_datasets = None
            self.input_constant_wave = None
            self.input_wave_count = None
            self.input_data_count = None
            self.input_chunk_count = None
            self.output_path = None
            self.output_dataset = None
            self.preload_arrays = False
            self.chunk_size = None

    def get_arg(self, name, old_value, args=None):
        args = args or self.args
        return util.args.get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        args = args or self.args
        return util.args.is_arg(name, args)

    def add_args(self, parser):
        parser.add_argument('--preload-arrays', action='store_true', help='Preload flux arrays into memory.\n')
        parser.add_argument('--chunk-size', type=int, help='Dataset chunk size.\n')

    def init_from_args(self, config, args):
        self.input_paths = []
        for path in args['in']:
            self.input_paths += glob.glob(path)

        self.output_path = args['out']
        
        self.preload_arrays = self.get_arg('preload_arrays', self.preload_arrays, args)
        self.chunk_size = self.get_arg('chunk_size', self.chunk_size, args)

    def init_inputs(self):
        # Open input datasets
        self.input_datasets = []
        self.input_data_count = 0
        self.input_chunk_count = 0

        for indir in self.input_paths:
            fn = os.path.join(indir, 'dataset.h5')
            ds = Dataset(preload_arrays=self.preload_arrays)
            ds.load(fn, format='h5')

            if self.input_wave_count is None:
                self.input_wave_count = ds.shape[1]
                self.input_constant_wave = ds.constant_wave
            elif self.input_wave_count != ds.shape[1]:
                raise Exception('Dataset wave counts are not equal.')
            elif self.input_constant_wave != ds.constant_wave:
                raise Exception('Some of the datasets have varying wave vectors.')

            self.input_data_count += ds.get_count()
            if self.chunk_size is not None:
                self.input_chunk_count += ds.get_chunk_count(self.chunk_size)

            self.input_datasets.append(ds)

    def init_output(self):
        fn = os.path.join(self.output_path, 'dataset.h5')
        ds = Dataset(preload_arrays=self.preload_arrays)
        ds.filename = fn
        ds.fileformat = 'h5'

        ds.init_storage(self.input_wave_count, self.input_data_count, constant_wave=self.input_constant_wave)
        
        self.output_dataset = ds

    def merge(self):
        if self.chunk_size is not None:
            t = tqdm(total=self.input_chunk_count)
            chunk_offset = 0
            for ds in self.input_datasets:
                chunk_count = ds.get_chunk_count(self.chunk_size)
                for chunk_id in range(chunk_count):
                    self.output_dataset.merge_chunk(ds, self.chunk_size, chunk_id, chunk_offset)
                    t.update(1)
                chunk_offset += chunk_count
            self.output_dataset.flush_cache_all(None, None)
        else:
            # TODO: implement in-memory merge
            raise NotImplementedError()

        # Save the dataset to store the wave vector, if necessary
        fn = os.path.join(self.output_path, 'dataset.h5')
        self.output_dataset.save(fn, format='h5')


import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..setup_logger import logger
from pfs.ga.pfsspec.core.scripts import Plugin
from .dataset import Dataset

class DatasetMerger(Plugin):
    def __init__(self, orig=None):
        super(DatasetMerger, self).__init__(orig=orig)

        if not isinstance(orig, DatasetMerger):
            self.preload_arrays = False
            self.chunk_size = None
            self.chunk_count = None
            self.labels = None
            self.input_paths = None
            self.input_datasets = None
            self.output_path = None
            self.output_dataset = None

            self.input_data_count = None
        else:
            self.preload_arrays = orig.preload_arrays
            self.chunk_size = orig.chunk_size
            self.chunk_count = orig.chunk_count
            self.labels = orig.labels
            self.input_paths = orig.input_paths
            self.input_datasets = orig.input_datasets
            self.output_path = orig.output_path
            self.output_dataset = orig.output_dataset

            self.input_data_count = orig.input_data_count

    def add_args(self, parser, config):
        super().add_args(parser, config)

        parser.add_argument('--preload-arrays', action='store_true', help='Preload flux arrays into memory.\n')
        parser.add_argument('--chunk-size', type=int, help='Dataset chunk size.\n')

        # A generic parameter to create extra dataset columns (labels) with python functions.
        parser.add_argument('--label', action='append', nargs=2, metavar=('name', 'function'), help='Additional dataset label from python expression.\n')

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

        self.preload_arrays = self.get_arg('preload_arrays', self.preload_arrays, args)
        self.chunk_size = self.get_arg('chunk_size', self.chunk_size, args)
        if self.chunk_size == 0:
            self.chunk_size = None

        # Additional labels
        self.labels = self.get_arg('label', self.labels, args)

    def create_dataset(self, preload_arrays=False):
        raise NotImplementedError()
    
    def open_input_data(self, input_paths):
        # Open each input dataset
        self.input_datasets = []
        self.input_data_count = None

        for path in input_paths:
            for p in glob.glob(path):
                fn = os.path.join(p, 'dataset.h5')
                ds = self.create_dataset(preload_arrays=self.preload_arrays)
                ds.load(fn)
                self.input_datasets.append(ds)

        for ds in self.input_datasets:
            if self.input_data_count is None:
                self.input_data_count = ds.get_count()
            elif self.input_data_count != ds.get_count():
                raise Exception('Dataset spectrum counts are not equal.')
            
            if self.chunk_size is not None and self.chunk_count is None:
                self.chunk_count = ds.get_chunk_count(self.chunk_size)
    
    def open_output_data(self, output_path):
        ds = self.create_dataset(preload_arrays=self.preload_arrays)
        ds.filename = os.path.join(output_path, 'dataset.h5')
        ds.fileformat = 'h5'

        self.output_dataset = ds

        # Allocate the output dataset in derived classes
    
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
            
            src = "lambda {}: {}".format(', '.join([c for c in self.output_dataset.params.columns]), exp)
            fun = eval(src)
            self.output_dataset.params[name] = self.output_dataset.params.apply(lambda r: fun(**r.to_dict()), axis=1)

    def process_item(self, i):
        # Implement to process the item i from the input datasets
        raise NotImplementedError()

    def merge(self):
        # Merge the params
        i = 0
        params = None
        for ds in self.input_datasets:
            if params is None:
                params = ds.params
            else:
                # Take the union of columns from each dataset. Use id as key.
                # Column names beginning with the second processed dataset will be suffixed with _2 _3, etc.
                params = pd.merge(params, ds.params, on='id', how='outer', suffixes=['', f'_{i}'])

            i += 1

        self.output_dataset.params = params
        self.output_dataset.save()

        if self.labels is not None:
            self.process_labels()

        if self.chunk_size is not None:
            # TODO: implement chunked merge
            raise NotImplementedError()
        else:
            for i in tqdm(range(self.input_data_count)):
                self.process_item(i)

        self.output_dataset.save()



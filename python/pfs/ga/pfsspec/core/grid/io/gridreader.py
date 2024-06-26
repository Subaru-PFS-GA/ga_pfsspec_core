import os
import numpy as np
from tqdm import tqdm
import multiprocessing

from ...setup_logger import logger
from pfs.ga.pfsspec.core.grid import GridEnumerator
from pfs.ga.pfsspec.core.util.smartparallel import SmartParallel
from pfs.ga.pfsspec.core.io import Importer

class GridReader(Importer):
    def __init__(self, grid=None, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, GridReader):
            self.args = None
            self.preload_arrays = False

            self.path = None
            self.grid = grid
        else:
            self.args = orig.args
            self.preload_arrays = orig.preload_arrays

            self.path = orig.path
            self.grid = grid if grid is not None else orig.grid

    def add_args(self, parser, config):
        super().add_args(parser, config)

        parser.add_argument('--preload-arrays', action='store_true', help='Preload arrays, do not use to save memory\n')

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)
        
        self.preload_arrays = self.get_arg('preload_arrays', self.preload_arrays, args)

    def create_grid(self):
        raise NotImplementedError()

    def get_array_grid(self):
        return self.grid

    def save_data(self, args, output_path):
        raise NotImplementedError()

    def read_grid(self, resume=False):
        if not resume:
            self.grid.allocate_values()

        # Iterate over the grid points and call a function for each
        logger.info("Reading grid {}.".format(type(self.grid).__name__))
        if self.top is not None:
            logger.info("Reading grid will stop after {} items.".format(self.top))

        grid = self.get_array_grid()
        slice = grid.get_slice()
        g = GridEnumerator(grid, s=slice, top=self.top, resume=resume)
        with SmartParallel(verbose=True, parallel=self.parallel, threads=self.threads) as p:
            for res in p.map(self.process_item, self.process_item_error, g):
                self.store_item(res)

        logger.info("Grid loaded.")

    def read_files(self, files, resume=False):
        if resume:
            # TODO: get ids from existing params data frame and remove the ones
            #       from the list that have already been imported
            raise NotImplementedError()

        # Iterate over a list of files and call a function for each
        logger.info("Loading {}".format(type(self.grid).__name__))
        if self.top is not None:
            logger.info("Loading will stop after {} spectra".format(self.top))
            files = files[:min(self.top, len(files))]

        k = 0
        t = tqdm(total=len(files))
        with SmartParallel(verbose=True, parallel=self.parallel, threads=self.threads) as p:
            for res in p.map(self.process_file, self.process_file_error, files):
                self.store_item(res)
                t.update(1)
                k += 1

        logger.info('{} files loaded.'.format(k))

    def process_item(self, i):
        raise NotImplementedError()
    
    def process_item_error(self, ex, i):
        raise NotImplementedError()

    def process_file(self, file):
        raise NotImplementedError()
    
    def process_file_error(self, ex, file):
        raise NotImplementedError()

    def store_item(self, res):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()
import os
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing

from pfs.ga.pfsspec.core.grid import GridEnumerator
from pfs.ga.pfsspec.core.util.smartparallel import SmartParallel
from pfs.ga.pfsspec.core.io import Importer

class GridReader(Importer):
    def __init__(self, grid=None, orig=None):
        super(GridReader, self).__init__(orig=orig)

        if not isinstance(orig, GridReader):
            self.args = None
            self.parallel = True
            self.threads = multiprocessing.cpu_count() // 2
            self.resume = False

            self.top = None
            self.preload_arrays = False

            self.path = None
            self.grid = grid
        else:
            self.args = orig.args
            self.parallel = orig.parallel
            self.threads = orig.threads
            self.resume = orig.resume

            self.top = orig.top
            self.preload_arrays = orig.preload_arrays

            self.grid = grid if grid is not None else orig.grid

    def add_args(self, parser):
        super(GridReader, self).add_args(parser)

        parser.add_argument('--top', type=int, default=None, help='Limit number of items.')
        parser.add_argument('--preload-arrays', action='store_true', help='Preload arrays, do not use to save memory\n')

    def init_from_args(self, config, args):
        super(GridReader, self).init_from_args(config, args)
        
        self.args = args

        self.top = self.get_arg('top', self.top, args)
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
        self.logger.info("Reading grid {}.".format(type(self.grid).__name__))
        if self.top is not None:
            self.logger.info("Reading grid will stop after {} items.".format(self.top))

        grid = self.get_array_grid()
        slice = grid.get_slice()
        g = GridEnumerator(grid, s=slice, top=self.top, resume=resume)
        t = tqdm(total=len(g))
        with SmartParallel(verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for res in p.map(self.process_item, g):
                self.store_item(res)
                t.update(1)

        self.logger.info("Grid loaded.")

    def read_files(self, files, resume=False):
        if resume:
            # TODO: get ids from existing params data frame and remove the ones
            #       from the list that have already been imported
            raise NotImplementedError()

        # Iterate over a list of files and call a function for each
        self.logger.info("Loading {}".format(type(self.grid).__name__))
        if self.top is not None:
            self.logger.info("Loading will stop after {} spectra".format(self.top))
            files = files[:min(self.top, len(files))]

        k = 0
        t = tqdm(total=len(files))
        with SmartParallel(verbose=True, parallel=self.parallel, threads=self.threads) as p:
            for res in p.map(self.process_file, files):
                self.store_item(res)
                t.update(1)
                k += 1

        self.logger.info('{} files loaded.'.format(k))

    def process_item(self, i):
        raise NotImplementedError()

    def process_file(self, file):
        raise NotImplementedError()

    def store_item(self, res):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()
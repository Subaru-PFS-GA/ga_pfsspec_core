import os
import numpy as np
import time
from tqdm import tqdm
import multiprocessing

from pfs.ga.pfsspec.core.scripts import Plugin

class GridBuilder(Plugin):
    def __init__(self, input_grid=None, output_grid=None, orig=None):
        super(GridBuilder, self).__init__(orig=orig)

        if isinstance(orig, GridBuilder):
            self.input_grid = input_grid if input_grid is not None else orig.input_grid
            self.output_grid = output_grid if output_grid is not None else orig.output_grid
            self.input_grid_index = None
            self.output_grid_index = None
            self.grid_shape = None
        else:

            self.input_grid = input_grid
            self.output_grid = output_grid
            self.input_grid_index = None
            self.output_grid_index = None
            self.grid_shape = None

    def add_args(self, parser, config):
        super().add_args(parser, config)

        # Axes of input grid can be used as parameters to filter the range
        grid = self.create_input_grid()
        grid.add_args(parser)

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

    def create_input_grid(self):
        raise NotImplementedError()

    def create_output_grid(self):
        raise NotImplementedError()

    def open_data(self, args, input_path, output_path):
        # Open and preprocess input
        if input_path is not None:
            self.input_grid = self.create_input_grid()
            self.open_input_grid(input_path)
            self.input_grid.init_from_args(args)
            self.input_grid.build_axis_indexes()

            self.grid_shape = self.input_grid.get_shape(s=self.input_grid.get_slice(), squeeze=False)

        # Open and preprocess output
        if output_path is not None:
            self.output_grid = self.create_output_grid()
            self.open_output_grid(output_path)

        self.build_data_index()
        self.verify_data_index()

    def build_data_index(self):
        raise NotImplementedError()

    def verify_data_index(self):
        # Make sure all data indices have the same shape
        assert(self.input_grid_index.shape[-1] == self.output_grid_index.shape[-1])

    def open_input_grid(self, input_path):
        raise NotImplementedError()

    def open_output_grid(self, output_path):
        raise NotImplementedError()

    def save_data(self, args, output_path):
        self.output_grid.save(self.output_grid.filename, format=self.output_grid.fileformat)

    def get_input_count(self):
        # Return the number of data vectors
        input_count = self.input_grid_index.shape[1]
        if self.top is not None:
            input_count = min(self.top, input_count)
        return input_count

    def init_process(self, worker_id):
        pass

    def run(self):
        raise NotImplementedError()

    def execute_notebooks(self, script):
        pass
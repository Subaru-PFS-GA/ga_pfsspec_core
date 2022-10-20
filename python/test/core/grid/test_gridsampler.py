import argparse
import os
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.grid import ArrayGrid, GridAxis, GridSampler

class TestArrayGrid(TestBase):
    def create_new_grid(self, preload_arrays=True):
        grid = ArrayGrid()
        grid.preload_arrays = preload_arrays

        grid.init_axis('a', np.array([1., 2., 3., 4., 5.]))
        grid.init_axis('b', np.array([10., 20., 30.]))
        grid.build_axis_indexes()

        grid.init_value('U')
        grid.init_value('V')

        return grid

    def init_empty_grid(self, grid):
        grid.init_value('U', (10,))
        grid.init_value('V', (100,))

        return grid

    def init_full_grid(self, grid):
        grid.init_value('U', (10,))
        grid.init_value('V', (100,))

        grid.values['U'] = np.random.rand(*grid.values['U'].shape)
        grid.values['V'] = np.random.rand(*grid.values['V'].shape)
        
        grid.build_value_indexes(rebuild=True)

        return grid

    def init_jagged_grid(self, grid):
        self.init_full_grid(grid)

        grid.values['U'][2:,2:,:] = np.nan
        grid.values['V'][2:,2:,:] = np.nan

        grid.build_value_indexes(rebuild=True)

        return grid

    def create_sampler(self, add_args=False):
        sampler = GridSampler()

        sampler.auxiliary_axes = {
            'c': GridAxis('c', order=0),
            'd': GridAxis('d', order=1),
        }

        if add_args:
            args = {
                'c': [ 10, 20, 21 ],
                'c_dist': [ 'normal', '0', '2' ],
                'd': [ 11, 12, 3 ],
                'd_dist': [ 'uniform' ],
            }
            sampler.init_from_args(None, args)
        
        return sampler

    def test_create(self):
        sampler = self.create_sampler()

    def test_add_args(self):
        sampler = self.create_sampler()
        parser = argparse.ArgumentParser()
        sampler.add_args(parser)
        
    def test_init_from_args(self):
        sampler = self.create_sampler(add_args=True)

    def test_get_auxiliary_gridpoint_count(self):
        sampler = self.create_sampler(add_args=True)
        c = sampler.get_auxiliary_gridpoint_count()
        self.assertEqual(21 * 3, c)

    def test_build_auxiliary_index(self):
        sampler = self.create_sampler(add_args=True)
        
        sampler.build_auxiliary_index()
        self.assertEqual((2, 63), sampler.auxiliary_index.shape)

    def test_draw_random_params(self):
        sampler = self.create_sampler(add_args=True)
        sampler.build_auxiliary_index()
        sampler.grid = self.create_new_grid()
        
        params = sampler.draw_random_params()

        self.assertEqual(4, len(params))

    def test_get_matched_params(self):
        sampler = self.create_sampler(add_args=True)
        sampler.match_params = pd.DataFrame({
            'id': [0, 1],
            'a': [1, 2],
            'b': [3, 4]
        })

        params = sampler.get_matched_params(0)
        self.assertEqual(3, len(params))

    def test_get_gridpoint_params(self):
        sampler = self.create_sampler(add_args=True)
        sampler.build_auxiliary_index()
        sampler.grid = self.create_new_grid()
        self.init_jagged_grid(sampler.grid)
        
        slice = sampler.grid.get_slice()
        index = sampler.grid.get_value_index_unsliced('U', s=slice)
        sampler.grid_index = np.array(np.where(index))

        params = sampler.get_gridpoint_params(0)
        self.assertEqual(4, len(params))
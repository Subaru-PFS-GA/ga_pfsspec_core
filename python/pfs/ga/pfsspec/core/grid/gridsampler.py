import numpy as np

from ..util.dist import get_random_dist
from ..util.args import *
from .grid import Grid, GridAxis

class GridSampler():
    """
    Base class to sample parameters on a grid. Implements function to scan a
    grid systematically or by randomly generating values within bounds. It also
    support matching parameters from an existing dataset.
    """

    def __init__(self, orig=None, random_state=None):

        if not isinstance(orig, GridSampler):
            self.random_state = random_state if random_state is not None else np.random

            self.grid = None                    # Stellar spectrum grid
            self.grid_index = None              # Valid models within grid

            self.auxiliary_axes = {}            # Additional axes that are not part of the grid but needs to be sampled or scanned systematically
            self.auxiliary_index = None

            self.match_params = None            # Match parameters from this dataset

            self.sample_mode = None             # Grid sampling mode (grid, random)
            self.sample_count = 0               # Number of samples to generate
                                                # TODO: move this elsewhere?
        else:
            self.random_state = random_state if random_state is not None else orig.random_state

            self.grid = orig.grid
            self.grid_index = None

            self.auxiliary_axes = orig.auxiliary_axes

            self.match_params = orig.match_params

            self.sample_mode = orig.sample_mode
            self.sample_dist = orig.sample_dist
            self.sample_count = orig.sample_count

    def init_auxiliary_axis(self, name, values=None, order=None):
        order = order or len(self.auxiliary_axes)
        axis = GridAxis(name, values, order=order)
        axis.build_index()
        self.auxiliary_axes[name] = axis

    def enumerate_auxiliary_axes(self, s=None, squeeze=False):
        if self.auxiliary_axes is not None:
            for i, k, ax in Grid.enumerate_axes_impl(self.auxiliary_axes, s=s, squeeze=squeeze):
                yield i, k, ax

    def add_args(self, parser):
        parser.add_argument('--sample-mode', type=str, choices=['grid', 'random'], default='grid', help='Sampling mode\n')
        parser.add_argument('--sample-count', type=int, default=None, help='Number of samples to be interpolated between models\n')

        # Additional axes
        for i, k, ax in self.enumerate_auxiliary_axes():
            ax.add_args(parser)

    def init_from_args(self, config, args):
        self.sample_mode = get_arg('sample_mode', self.sample_mode, args)
        self.sample_count = get_arg('sample_count', self.sample_count, args)

        # Register command-line arguments for auxiliary axes
        for i, k, ax in self.enumerate_auxiliary_axes():
            # Using aux mode because we need to generate the axis values from the
            # parameters passed to via args.
            ax.init_from_args(args, mode='aux')

    def get_gridpoint_count(self):
        self.grid_index.shape[1] * self.auxiliary_index.shape[1]

    def get_auxiliary_gridpoint_count(self):
        """
        Count the number of auxiliary grid points
        """
        count = 1
        for p in self.auxiliary_axes:
            if self.auxiliary_axes[p] is not None:
                count *= self.auxiliary_axes[p].values.size
        return count

    def build_auxiliary_index(self):
        count = 0
        size = 1
        shape = ()
        for j, p, ax in self.enumerate_auxiliary_axes():
            ax.build_index()
            count += 1
            size *= ax.values.size
            shape = shape + (ax.values.size,)
        self.auxiliary_index = np.indices(shape).reshape(count, size)

    def draw_random_params(self):
        """
        Draw random values for each of the axes, those of the grid as well as
        auxiliary.
        """
        
        # Always draw random parameters from self.random_state
        def draw_random_param(axis):
            dist = get_random_dist(axis.dist, self.random_state)
            if dist is not None:
                # Rescale values between the limits
                r = dist() if axis.dist_args is None else dist(*axis.dist_args)
                return axis.min + r * (axis.max - axis.min)

        # Draw from grid parameters
        params = {}
        s = self.grid.get_slice()
        for i, p, axis in self.grid.enumerate_axes(s=s, squeeze=False):
            params[p] = draw_random_param(axis)
            
        # Draw from auxiliary parameters
        for i, p, axis in self.enumerate_auxiliary_axes():
            params[p] = draw_random_param(axis)

        return params

    def get_matched_params(self, i):
        # When ˙match_params` is set to a DataFrame, we need to look up the ith
        # record and return the row as a dictionary
        params = self.match_params[self.match_params['id'] == i].to_dict('records')[0]
        return params

    def get_gridpoint_params(self, i):
        idx = tuple(self.grid_index[:, i % self.grid_index.shape[1]])

        # Look up parameters in the underlying grid
        params = {}
        s = self.grid.get_slice()
        for j, p, axis in self.grid.enumerate_axes(s=s):
            params[p] = axis.values[self.grid_index[j, i % self.grid_index.shape[1]]]

        # Append parameters from the auxiliary axes
        i = i // self.grid_index.shape[1]
        for j, p, axis in self.enumerate_auxiliary_axes():
            params[p] = axis.values[self.auxiliary_index[j, i]]

        return params
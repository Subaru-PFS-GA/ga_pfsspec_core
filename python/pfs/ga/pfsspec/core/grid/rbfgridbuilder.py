import os
import gc
import numpy as np
import time
from tqdm import tqdm

from ..util.array import *
from ..util.interp import Rbf
from .grid import Grid
from .gridbuilder import GridBuilder
from .arraygrid import ArrayGrid

class RbfGridBuilder(GridBuilder):
    def __init__(self, input_grid=None, output_grid=None, orig=None):
        super(RbfGridBuilder, self).__init__(input_grid=input_grid, output_grid=output_grid, orig=orig)

        if isinstance(orig, RbfGridBuilder):
            self.padding = orig.padding
            self.fill = orig.fill
            self.interpolation = orig.interpolation
            self.function = orig.function
            self.epsilon = orig.epsilon
            self.smoothing = orig.smoothing
            self.method = orig.method
        else:
            self.padding = False
            self.fill = True
            self.interpolation = 'ijk'
            self.function = 'multiquadric'
            self.epsilon = None
            self.smoothing = 0.0
            self.method = None

    def add_args(self, parser, config):
        super(RbfGridBuilder, self).add_args(parser, config)
        parser.add_argument('--padding', action='store_true', help='Pad array by one prior to RBF.\n')
        parser.add_argument('--fill', dest='fill', action='store_true', help='Fill in masked points.\n')
        parser.add_argument('--no-fill', dest='fill', action='store_false', help='Do not fill in masked points.\n')
        parser.add_argument('--interpolation', type=str, default='multiquadric', choices=['ijk', 'xyz'],
            help='Interpolation in array index or axis values.\n')
        parser.add_argument('--function', type=str, default='multiquadric', 
            choices=['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'],
            help='RBF kernel function.\n')
        parser.add_argument('--epsilon', type=float, help='Adjustable constant for Gaussian and multiquadric.\n')
        parser.add_argument('--smoothing', type=float, help='RBF smoothing coeff.\n')
        parser.add_argument('--method', type=str, default=None, choices=['sparse', 'solve', 'nnls', 'skip'])

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

        self.padding = self.get_arg('padding', self.padding, args)
        self.fill = self.get_arg('fill', self.fill, args)
        self.function = self.get_arg('function', self.function, args)
        self.epsilon = self.get_arg('epsilon', self.epsilon, args)
        self.smoothing = self.get_arg('smoothing', self.smoothing, args)
        self.method = self.get_arg('method', self.method, args)

    def fit_rbf(self, value, axes, mask=None, method=None, function=None, epsilon=None, callback=None):
        """Returns the Radial Base Function interpolation of a grid slice.

        Args:
            value
            axes
            mask (array): Mask, must be the same shape as the grid.
            function (str): Basis function, see RBF documentation.
            epsilon (number): See RBF documentation.
            smooth (number): See RBF documentation.
            method (string): use solve or nnls to fit data
        """

        naxes = len(axes)

        # Since we must have the same number of grid points as unmasked elements,
        # we need to contract the mask along all value array dimensions that are
        # not along the axes. Since `value` is already squeezed, only use axes
        # that do not match axes in kwargs.
        m = ~np.isnan(value)
        if len(m.shape) > naxes:
            m = np.all(m, axis=-(len(m.shape) - naxes))

        # We assume that the provided mask has the same shape as the grid
        if mask is not None:
            m &= mask
            
        m = m.flatten()

        # Only use the first ˙self.top` grid points
        if self.top is not None:
            c = np.cumsum(m)
            t = np.where(c == self.top)
            if t[0].shape[0] > 0:
                t = t[0][0]
                m[t + 1:] = False

        # Flatten slice along axis dimensions
        sh = 1
        for i in range(naxes):
            sh *= value.shape[i]
        value = value.reshape((sh,) + value.shape[naxes:])
        value = value[m]

        # Get the grid points along each axis. Padding must be False here because
        # we don't want to shift the grid indexes to calculate the RBF, otherwise the
        # index would need to be stored in the file to know if it is a padded grid.
        all_points = ArrayGrid.get_meshgrid_points(axes, padding=False, squeeze=True, interpolation=self.interpolation, indexing='ij')
        
        # Generate the grid from the axis points and apply the mask.
        all_points = [ all_points[p].flatten() for i, p, ax in Grid.enumerate_axes_impl(axes, squeeze=True) ]
        points = [ p[m] for p in all_points]

        # points: list of arrays of shape of (unmasked_count,), for each non-contracted axis
        # value: shape: (unmasked_count, value_dim)

        if len(value.shape) == 1:
            mode = '1-D'
        else:
            mode = 'N-D'

        rbf = Rbf()
        rbf.weight_matrix_callback = callback
        rbf.fit(*points, value, function=function or self.function,
            epsilon=epsilon or self.epsilon, smooth=self.smoothing,
            mode=mode, method=self.method or method)

        if self.fill:
            # Fill in the results with zeros at masked grid points so that
            # the resulting RBF grids of the same size can be constructed at the end

            rbf.xi = np.stack(all_points, axis=0)
            
            nodes = np.zeros((rbf.xi.shape[1], rbf.nodes.shape[1]), dtype=rbf.nodes.dtype)
            nodes[m, :] = rbf.nodes
            rbf.nodes = nodes

        gc.collect()

        return rbf
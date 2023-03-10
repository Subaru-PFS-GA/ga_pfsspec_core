import logging
import numpy as np

from pfs.ga.pfsspec.core import PfsObject
from pfs.ga.pfsspec.core.grid import Grid

class PcaGrid(PfsObject):
    # Wraps an ArrayGrid or an RbfGrid and adds PCA decompression support

    PCA_TRANSFORM_FUNCTIONS = {
        'sqrt': {
            'forward': lambda x: np.sign(x) * np.sqrt(np.sign(x) *x),
            'backward': lambda x: np.sign(x) * x**2,
            'backward-error': lambda x, s2: 2 * x**2 * s2
        },
        'asinhsqrt': {
            'forward': lambda x: np.arcsinh(np.sign(x) * np.sqrt(np.sign(x) *x)),
            'backward': lambda x: np.sign(x) * np.sinh(x)**2
        },
        'square': {
            'forward': lambda x: np.sign(x) * x**2,
            'backward': lambda x: np.sign(x) * np.sqrt(np.sign(x) *x)
        },
        'log': {
            'forward': np.log,
            'backward': np.exp
        },
        'exp': {
            'forward': np.exp,
            'backward': np.log
        },
        'safelog': {
            'forward': lambda x: np.log(np.where(x < 1e-10, 1e-10, x)), 
            'backward': lambda x: np.exp(np.where(x > 1e2, 1e2, x))
        }
    }

    POSTFIX_PCA = 'pca'
    POSTFIX_EIGS = 'eigs'
    POSTFIX_EIGV = 'eigv'
    POSTFIX_MEAN = 'mean'
    POSTFIX_ERROR = 'error'
    POSTFIX_TRANSFORM = 'transform'

    def __init__(self, grid, orig=None):
        super(PcaGrid, self).__init__(orig=orig)

        if not isinstance(orig, PcaGrid):
            self.grid = grid

            self.eigs = {}
            self.eigv = {}
            self.mean = {}
            self.error = {}
            self.transform = {}
            self.k = None
        else:
            self.grid = grid if grid is not None else orig.grid

            self.eigs = orig.eigs
            self.eigv = orig.eigv
            self.mean = orig.mean
            self.error = orig.error
            self.transform = orig.transform
            self.k = orig.k

    @property
    def preload_arrays(self):
        return self.grid.preload_arrays

    @preload_arrays.setter
    def preload_arrays(self, value):
        self.grid.preload_arrays = value

    @property
    def array_grid(self):
        return self.grid.array_grid

    @property
    def pca_grid(self):
        return self

    @property
    def rbf_grid(self):
        return self.grid.rbf_grid

    def init_from_args(self, args):
        self.grid.init_from_args(args)

    def get_slice(self):
        """
        Returns the slicing of the grid.
        """
        return self.grid.get_slice()

    def get_shape(self, s=None, squeeze=False):
        return self.grid.get_shape(s=s, squeeze=squeeze)

    def get_constants(self):
        return self.grid.get_constants()

    def get_value_path(self, name):
        return '/'.join([Grid.PREFIX_GRID, Grid.PREFIX_ARRAYS, name, self.POSTFIX_PCA])

    def set_constants(self, constants):
        self.grid.set_constants(constants)

    def init_axes(self):
        raise NotImplementedError()

    def init_axis(self, name, values=None):
        self.grid.init_axis(name, values=values)

    def set_axes(self, axes):
        self.grid.set_axes(axes)

    def get_axis(self, key):
        return self.grid.get_axis(key)

    def enumerate_axes(self, s=None, squeeze=False):
        return self.grid.enumerate_axes(s=s, squeeze=squeeze)

    def build_axis_indexes(self):
        self.grid.build_axis_indexes()

    def init_constant(self, name):
        self.grid.init_constant(name)

    def get_constant(self, name):
        return self.grid.get_constant(name)

    def set_constant(self, name, value):
        self.grid.set_constant(name, value)

    def init_value(self, name, shape=None, pca=False, **kwargs):
        if pca:
            if shape is not None:
                # Last dimension is number of coefficients
                self.grid.init_value(name, shape=shape[-1])
            else:
                self.grid.init_value(name)
            self.eigs[name] = None
            self.eigv[name] = None
            self.mean[name] = None
            self.error[name] = None
        else:
            self.grid.init_value(name, shape=shape)

    def allocate_values(self):
        raise NotImplementedError()

    def allocate_value(self, name, shape=None, pca=False):
        if pca:
            if shape is not None:
                self.grid.allocate_value(name, shape=(shape[-1],))
                # TODO: there should be some trickery here regarding
                #       the size of these arrays
                self.eigs[name] = np.full(shape[:-1], np.nan)
                self.eigv[name] = np.full(shape[:-1], np.nan)
                self.mean[name] = np.full(shape[:-1], np.nan)
                self.error[name] = np.full(shape[:-1], np.nan)
            else:
                self.grid.allocate_value(name)
                self.eigs[name] = None
                self.eigv[name] = None
                self.mean[name] = None
                self.error[name] = None
        else:
            self.grid.allocate_value(name, shape=shape)

    def add_args(self, parser):
        self.grid.add_args(parser)

        parser.add_argument('--pca-truncate', type=int, default=None, help='PCA truncate.')

    def init_from_args(self, args):
        self.grid.init_from_args(args)

        self.k = self.get_arg('pca_truncate', self.k, args)

    def get_index(self, **kwargs):
        return self.grid.get_index(**kwargs)

    def get_nearest_index(self, **kwargs):
        return self.grid.get_nearest_index(**kwargs)

    def has_value_index(self, name):
        return self.grid.has_value_index(name)

    def get_valid_value_count(self, name, s=None):
        return self.grid.get_valid_value_count(name, s=s)

    def has_value(self, name):
        return self.grid.has_value(name)

    def has_error(self, name):
        return self.has_value(name) and self.error[name] is not None

    def has_value_at(self, name, idx, mode='any'):
        return self.grid.has_value_at(name, idx, mode=mode)

    def set_value(self, name, value, s=None, **kwargs):
        # TODO: slicing?
        self.eigs[name] = kwargs.pop('eigs', None)
        self.eigv[name] = kwargs.pop('eigv', None)
        self.mean[name] = kwargs.pop('mean', None)
        self.error[name] = kwargs.pop('error', None)
        self.transform[name] = kwargs.pop('transform', None)

        self.grid.set_value(name, value, s=s, **kwargs)
        
    def get_value_at(self, name, idx, s=None, raw=False):
        if not raw and name in self.eigs:
            s = s or slice(None)

            pc = self.grid.get_value_at(name, idx)
            if self.k is None:
                v = np.dot(self.eigv[name][s], pc)
            else:
                v = np.dot(self.eigv[name][s][:, :self.k], pc[:self.k])

            # Add mean
            if self.mean[name] is not None:
                v += self.mean[name][s]

            # Inverse transform
            if self.transform[name] is not None and self.transform[name] != 'none':
                v = PcaGrid.PCA_TRANSFORM_FUNCTIONS[self.transform[name]]['backward'](v)

            return v
        else:
            return self.grid.get_value_at(name, idx, s=s)

    def get_error_at(self, name, idx, s=None, raw=None):
        """Return the error associated to the value `name`."""

        s = s or slice(None)

        # Error is the same everywhere over the grid for PCA, that's how it works
        error = self.error[name][s]

        # Propage uncertainty through the transformation
        if self.transform[name] is not None and self.transform[name] != 'none':
            error = PcaGrid.PCA_TRANSFORM_FUNCTIONS[self.transform[name]]['backward-error'](error)

        return error

    def get_value(self, name, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_value_at(name, idx, s=s)

    def get_values_at(self, idx, s=None, names=None):
        names = names or self.values.keys()
        return {name: self.get_value_at(name, idx, s) for name in names}

    def get_values(self, s=None, names=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_values_at(idx, s, names=names)

    def get_chunk_shape(self, name, shape, s=None):
        # Override chunking for eigv to be able to memory map directly from
        # the file. This is necessary to run on multiple threads.
        
        if name.endswith(self.POSTFIX_EIGV):
            return None
        else:
            return super().get_chunk_shape(name, shape, s)

    def save_items(self):
        self.grid.filename = self.filename
        self.grid.fileformat = self.fileformat
        self.grid.save_items()

        for name in self.eigs:
            if self.eigs[name] is not None:
                path = self.get_value_path(name)
                self.save_item('/'.join([path, self.POSTFIX_EIGS]), self.eigs[name])
                self.save_item('/'.join([path, self.POSTFIX_EIGV]), self.eigv[name])
                self.save_item('/'.join([path, self.POSTFIX_MEAN]), self.mean[name])
                self.save_item('/'.join([path, self.POSTFIX_ERROR]), self.error[name])
                self.save_item('/'.join([path, self.POSTFIX_TRANSFORM]), self.transform[name])

    def load_items(self, s=None):
        self.grid.filename = self.filename
        self.grid.fileformat = self.fileformat
        self.grid.load_items(s=s)

        for name in self.eigs:
            path = self.get_value_path(name)

            if self.k is not None:
                pca_slice = np.s_[..., :self.k]
            else:
                pca_slice = None

            self.eigs[name] = self.load_item('/'.join([path, self.POSTFIX_EIGS]), np.ndarray, s=pca_slice)
            self.eigv[name] = self.load_item('/'.join([path, self.POSTFIX_EIGV]), np.ndarray, s=pca_slice, mmap=True)
            self.mean[name] = self.load_item('/'.join([path, self.POSTFIX_MEAN]), np.ndarray)
            self.error[name] = self.load_item('/'.join([path, self.POSTFIX_ERROR]), np.ndarray)
            self.transform[name] = self.load_item('/'.join([path, self.POSTFIX_TRANSFORM]), str)

    def set_object_params(self, obj, idx=None, **kwargs):
        self.grid.set_object_params(obj, idx=idx, **kwargs)
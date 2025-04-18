import numpy as np

from ..setup_logger import logger
from pfs.ga.pfsspec.core.util.interp import Rbf
from .grid import Grid

class RbfGrid(Grid):
    # Implements a grid that supports interpolation based on RBF. This is not
    # necessarily a grid, as RBF interpolation extends function to any point
    # but nodes are defined based on the predefined values along the axes.

    POSTFIX_RBF = 'rbf'
    POSTFIX_XI = 'xi'
    POSTFIX_NODES = 'a'
    POSTFIX_C = 'c'
    POSTFIX_FUNCTION = 'function'
    POSTFIX_EPSILON = 'eps'

    def __init__(self, config=None, orig=None):
        super(RbfGrid, self).__init__(orig=orig)

        if isinstance(orig, RbfGrid):
            self.config = config if config is not None else orig.config
            self.preload_arrays = orig.preload_arrays
            self.values = orig.values
            self.value_shapes = orig.value_shapes
        else:
            self.config = config
            self.preload_arrays = False
            self.values = {}
            self.value_shapes = {}

            self.init_values()

    @property
    def array_grid(self):
        return None

    @property
    def rbf_grid(self):
        return self

    @property
    def pca_grid(self):
        return None
    
    def init_from_args(self, args, slice_from_args=True):
        super().init_from_args(args)

    def ensure_lazy_load(self):
        # This works with HDF5 format only!
        if not self.preload_arrays and self.fileformat != 'h5':
            raise NotImplementedError()

    def get_value_path(self, name):
        # This could be extended to be backward compatible with older versions of the rbf grid
        return '/'.join([Grid.PREFIX_GRID, Grid.PREFIX_ARRAYS, name, self.POSTFIX_RBF])

    def init_value(self, name, shape=None, **kwargs):
        if shape is None:
            self.values[name] = None
            self.value_shapes[name] = shape
        else:
            if len(shape) != 1:
                raise NotImplementedError()

            self.value_shapes[name] = shape

            valueshape = self.xi.shape[:1] + tuple(shape)

            if self.preload_arrays:
                logger.info('Initializing memory for RBF "{}" of size {}...'.format(name, valueshape))
                self.values[name] = np.full(valueshape, np.nan)
                logger.info('Initialized memory for RBF "{}" of size {}.'.format(name, valueshape))
            else:
                self.values[name] = None
                logger.info('Initializing data file for RBF "{}" of size {}...'.format(name, valueshape))
                if not self.has_item(name):
                    self.allocate_item(name, valueshape, dtype=float)
                logger.info('Skipped memory initialization for RBF "{}". Will read random slices from storage.'.format(name))

    def allocate_value(self, name, shape=None):
        if shape is not None:
            self.value_shapes[name] = shape
        self.init_value(name, self.value_shapes[name])

    def has_value_index(self, name):
        # RBF doesn't have an index, it's continuous
        return False

    def get_nearest_index(self, **kwargs):
        # RBF doesn't have an index per se, but we can return the interpolated
        # grid values from physical values
        return self.get_index(**kwargs)

    def get_index(self, **kwargs):
        # Convert values to axis index, the input to RBF
        
        # RBF works a little bit differently when a dimension is 1 length than
        # normal arrays because even though that dimension is not squeezed, the
        # RBF indexes are.
        
        # Only include index along the axis if it's not a 1-length axis
        idx = ()
        for i, p, axis in self.enumerate_axes(squeeze=True):
            if p in kwargs:
                idx += (axis.ip_to_index(kwargs[p]),)
            else:
                idx += (None,)

        return idx
    
    def get_params_at(self, idx=None, **kwargs):
        params = {}

        if kwargs is not None:
            params = { k: v for k, v in kwargs.items()}

        if idx is not None:
            for i, p, axis in self.enumerate_axes():
                if idx[i] is not None:
                    params[p] = float(axis.ip_to_value(idx[i]))
                else:
                    params[p] = None

        return params

    def has_value(self, name):
        if self.preload_arrays:
            return name in self.values and self.values[name] is not None
        else:
            return name in self.values and self.has_item(name + '/rbf/xi')

    def has_error(self, name):
        # TODO: implement error propagation if error is available
        return False

    def set_values(self, values, s=None, **kwargs):
        for k in values:
            self.set_value(k, values[k], s=s, **kwargs)

    def set_value(self, name, value, s=None, **kwargs):
        if s is not None or len(kwargs) > 0:
            raise Exception('RbfGrid does not support slicing and indexing.')
        if not isinstance(value, Rbf):
            raise Exception('Value must be an Rbf object.')
        self.values[name] = value

    def has_value_at(self, name, idx, mode='any'):
        return True

    def check_extrapolation(self, idx):
        # Only include index along the axis if it's not a 1-length axis
        for i, k, axis in self.enumerate_axes(squeeze=True):
            if np.any((idx[i] < 0) | (axis.values.shape[0] < idx[i])):
                return True
        return False

    def get_value_at(self, name, idx, s=None, pre_process=None, cache_key_prefix=(), extrapolate=False):
        idx = Grid.rectify_index(idx)
        
        if not extrapolate and self.check_extrapolation(idx):
            logger.warning("Requested point of RBF grid would result in extrapolation")
            return None
        
        value = self.values[name](*idx)
        
        if value is not None:
            if pre_process is not None:
                value = pre_process(value)

            value = value[s or ()]

        # NOTE: RBF cannot cache values

        return value

    def get_error_at(self, name, idx):
        """Return the error associated to the value `name`."""

        # TODO: implement
        raise NotImplementedError()

    def get_value(self, name, s=None, extrapolate=False, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_value_at(name, idx, s=s, extrapolate=extrapolate)

    def get_values_at(self, idx, s=None, names=None, extrapolate=False):
        # Return all values interpolated to idx. This function is optimized for
        # RBF grids where the distance matrix computation is expensive.
        idx = Grid.rectify_index(idx)       
        names = names or self.values.keys()

        if not extrapolate and self.check_extrapolation(idx):
            return None
        
        # We can potentially reuse the distance/kernel matrix if the grid points
        # and the kernels are the same
        values = {}
        A = None
        prev = None
        for name in names:
            if prev is None or not prev.is_compatible(self.values[name]):
                prev = self.values[name]
                y, A = self.values[name].eval(*idx)
            else:
                y, A = self.values[name].eval(*idx, A=A)
            values[name] = y[s or ()]

        return values

    def get_values(self, s=None, names=None, extrapolate=False, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_values_at(idx, s, names=names, extrapolate=extrapolate)

    def get_chunk_shape(self, name, shape, s=None):
        # Override chunking for the nodes to enable mmapping the large array
        if name.endswith(self.POSTFIX_NODES):
            return None
        else:
            return super().get_chunk_shape(name, shape, s=s)

    def save_items(self):
        super(RbfGrid, self).save_items()
        self.save_values()

    def save_values(self):
        # TODO: implement chunking along the value dimensions if necessary
        # TODO: this is a little bit redundant here because we store the xi values
        #       for each RBF. The coordinates are supposed to be the same for
        #       all data values.
        for name in self.values:
            if self.values[name] is not None:
                logger.info('Saving RBF "{}" of size {}'.format(name, self.values[name].nodes.shape))

                path = self.get_value_path(name)
                self.save_item('/'.join([path, self.POSTFIX_XI]), self.values[name].xi)
                self.save_item('/'.join([path, self.POSTFIX_NODES]), self.values[name].nodes)
                self.save_item('/'.join([path, self.POSTFIX_C]), self.values[name].c)
                self.save_item('/'.join([path, self.POSTFIX_FUNCTION]), self.values[name].function)
                self.save_item('/'.join([path, self.POSTFIX_EPSILON]), self.values[name].epsilon)

                logger.info('Saved RBF "{}" of size {}'.format(name, self.values[name].nodes.shape))

    def load_items(self, s=None):
        super(RbfGrid, self).load_items(s=s)
        self.load_values(s=s)

    def load_rbf(self, xi, nodes, c, function='multiquadric', epsilon=None):
        # TODO: bring out kernel function name as parameter or save into hdf5
        rbf = Rbf()
        rbf.load(nodes, xi, c, function=function, epsilon=epsilon, mode='N-D')
        return rbf

    def load_values(self, s=None):
        # TODO: implement chunked lazy loading along the value dimension, if necessary
        if s is not None:
            raise NotImplementedError()

        for name in self.values:
            logger.info('Loading RBF "{}" of size {}'.format(name, s))

            path = self.get_value_path(name)
            xi = self.load_item('/'.join([path, self.POSTFIX_XI]), np.ndarray)
            nodes = self.load_item('/'.join([path, self.POSTFIX_NODES]), np.ndarray, mmap=True, skip_mmap=True)
            c = self.load_item('/'.join([path, self.POSTFIX_C]), np.ndarray)
            function = self.load_item('/'.join([path, self.POSTFIX_FUNCTION]), str)
            epsilon = self.load_item('/'.join([path, self.POSTFIX_EPSILON]), float)
            if xi is not None and nodes is not None:
                self.values[name] = self.load_rbf(xi, nodes, c, function=function, epsilon=epsilon)
                logger.info('Loaded RBF "{}" of size {}'.format(name, s))
            else:
                self.values[name] = None
                logger.info('Skipped loading RBF "{}" of size {}'.format(name, s))
            
    def set_object_params_idx(self, obj, idx):
        # idx might be squeezed, use single values along those axes
        i = 0
        for j, p, axis in self.enumerate_axes():
            if axis.values.shape[0] > 1:
                setattr(obj, p, float(axis.ip_to_value(idx[i])))
                i += 1
            else:
                setattr(obj, p, float(axis.values[0]))

    def interpolate_value_rbf(self, name, s=None, pre_process=None, cache_key_prefix=(), extrapolate=False, **kwargs):
        idx = self.get_index(**kwargs)
        value = self.get_value_at(name, idx, s=s, pre_process=pre_process, cache_key_prefix=cache_key_prefix, extrapolate=extrapolate)
        return value, kwargs
    
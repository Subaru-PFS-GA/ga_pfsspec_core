import os
import logging
import numpy as np
import itertools
from collections.abc import Iterable
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, CubicSpline
from scipy.interpolate import interp1d, interpn

from ..util.array import *
from .grid import Grid
from .gridaxis import GridAxis

# TODO: add support for value dtype as in Dataset
class ArrayGrid(Grid):
    """Implements a generic grid class to store and interpolate data.

    This class implements a generic grid class with an arbitrary number of axes
    and multiple value arrays in each grid point. Value arrays must have the same
    leading dimensions as the size of the axes. An index is build on every
    value array which indicated valid/invalid data in a particular grid point.
    """

    POSTFIX_VALUE = 'value'
    POSTFIX_INDEX = 'index'

    def __init__(self, config=None, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, ArrayGrid):
            self.config = config        # Grid configuration class
            self.preload_arrays = False # Preload arrays into memory
            self.mmap_arrays = False    # MMap arrays into memory from HDF5
            self.values = {}            # Dictionary of value arrays
            self.value_shapes = {}      # Dictionary of value array shapes (excl. grid shape)
            self.value_indexes = {}     # Dictionary of index array indicating existing grid points
            self.slice = None           # Global mask defined as a slice

            self.value_cache = None

            self.init_values()
        else:
            self.config = config if config is not None else orig.config
            self.preload_arrays = orig.preload_arrays
            self.mmap_arrays = orig.mmap_arrays
            self.values = orig.values
            self.value_shapes = orig.value_shapes
            self.value_indexes = orig.value_indexes
            self.slice = orig.slice

            self.value_cache = orig.value_cache

    @property
    def array_grid(self):
        return self

    @property
    def rbf_grid(self):
        return None

    @property
    def pca_grid(self):
        return None

#region Support slicing via command-line arguments

    def init_from_args(self, args, slice_from_args=True):
        super().init_from_args(args)
        if slice_from_args:
            self.slice = self.get_slice_from_args(args)

    def get_slice_from_args(self, args):
        # If a limit is specified on any of the parameters on the command-line,
        # try to slice the grid while loading from HDF5
        s = []
        for i, k, ax in self.enumerate_axes():
            if k in args and args[k] is not None:
                values = args[k]
                if not isinstance(values, Iterable):
                    values = [ values ]

                if len(values) == 2:
                    idx = np.digitize([values[0], values[1]], ax.values)
                    s.append(slice(max(0, idx[0] - 1), idx[1], None))
                elif len(values) == 1:
                    idx = np.digitize([values[0]], ax.values)
                    s.append(max(0, idx[0] - 1))
                else:
                    raise Exception('Only two or one values are allowed for parameter {}'.format(k))
            else:
                s.append(slice(None))

        return tuple(s)

    def ensure_lazy_load(self):
        # This works with HDF5 format only!
        if not (self.preload_arrays or self.mmap_arrays) and self.fileformat != 'h5':
            raise NotImplementedError()

    def get_slice(self):
        """
        Returns the slicing of the grid.
        """
        return self.slice

    def get_shape(self, s=None, squeeze=False):
        shape = tuple(axis.values.shape[0] for i, p, axis in self.enumerate_axes(s=s, squeeze=squeeze))
        return shape
        
    def get_value_path(self, name):
        return '/'.join([self.PREFIX_GRID, self.PREFIX_ARRAYS, name, self.POSTFIX_VALUE])

    def get_index_path(self, name):
        return '/'.join([self.PREFIX_GRID, self.PREFIX_ARRAYS, name, self.POSTFIX_INDEX])

#endregion

    def get_value_shape(self, name, s=None):
        # Gets the full shape of the grid. It assumes different last
        # dimensions for each data array.

        if s is not None:
            # Allow slicing of values, this requires some calculations
            raise NotImplementedError()
        else:
            shape = self.get_shape(s=self.slice, squeeze=False) + self.value_shapes[name]

        return shape

    # TODO: add support for dtype
    def init_value(self, name, shape=None, **kwargs):
        if shape is None:
            self.values[name] = None
            self.value_shapes[name] = None
            self.value_indexes[name] = None
        else:
            if len(shape) != 1:
                raise NotImplementedError()

            grid_shape = self.get_shape(s=self.slice, squeeze=False)
            value_shape = grid_shape + tuple(shape)

            self.value_shapes[name] = shape

            if self.preload_arrays:
                self.logger.info('Initializing memory for grid "{}" of size {}...'.format(name, value_shape))
                self.values[name] = np.full(value_shape, np.nan)
                self.value_indexes[name] = np.full(grid_shape, False, dtype=bool)
                self.logger.info('Initialized memory for grid "{}" of size {}.'.format(name, value_shape))
            elif self.mmap_arrays:
                # TODO: allocate HDF5 dataset then mmap it
                raise NotImplementedError()
            else:
                self.values[name] = None
                self.value_indexes[name] = None
                self.logger.info('Initializing data file for grid "{}" of size {}...'.format(name, value_shape))
                if not self.has_item(self.get_value_path(name)):
                    self.allocate_item(self.get_value_path(name), value_shape, dtype=float)
                    self.allocate_item(self.get_index_path(name), grid_shape, dtype=bool)
                self.logger.info('Skipped memory initialization for grid "{}". Will read random slices from storage.'.format(name))

    def allocate_value(self, name, shape=None):
        if shape is not None:
            self.value_shapes[name] = shape
        self.init_value(name, self.value_shapes[name])

    def is_value_valid(self, name, value):
        if self.config is not None:
            return self.config.is_value_valid(self, name, value)
        else:
            return np.logical_not(np.any(np.isnan(value), axis=-1))

    def build_value_indexes(self, rebuild=False):
        for name in self.values:
            self.build_value_index(name, rebuild=rebuild)

    def build_value_index(self, name, rebuild=False):
        if rebuild and not self.preload_arrays:
            raise Exception('Cannot build index on lazy-loaded grid.')
        elif rebuild or name not in self.value_indexes or self.value_indexes[name] is None:
            self.logger.debug('Building indexes on grid "{}" of size {}'.format(name, self.value_shapes[name]))
            self.value_indexes[name] = self.is_value_valid(name, self.values[name])
        else:
            self.logger.debug('Skipped building indexes on grid "{}" of size {}'.format(name, self.value_shapes[name]))
        self.logger.debug('{} valid vectors in grid "{}" found'.format(np.sum(self.value_indexes[name]), name))

    def get_valid_value_count(self, name, s=None):
        return np.sum(self.get_value_index(name, s=s))

    def is_on_grid(self, **kwargs):
        """
        Returns true if the parameters mark a valid grid position. It doesn't
        take the grid mask into account but the grid point must be inside the
        bounds of the grid.
        """

        for i, p, axis in self.enumerate_axes():
            if p in kwargs and kwargs[p] not in axis.values:
                return False

        return True
           
    def get_index(self, **kwargs):
        """Returns the indexes along all axes corresponding to the values specified.

        If an axis name is missing, a whole slice is returned.

        Returns:
            [type]: [description]
        """

        idx = ()
        for i, p, axis in self.enumerate_axes():
            if p in kwargs:
                idx += (axis.index[kwargs[p]],)
            else:
                if self.slice is None:
                    idx += (slice(None),)
                else:
                    idx += (self.slice[i],)

        return idx

    def get_nearest_index(self, **kwargs):
        """
        Returns the nearest indexes along all axes correspondint to the values
        specified in `kwargs`.

        If an axis name is missing, a whole slice is returned.
        """

        idx = ()
        for i, p, axis in self.enumerate_axes():
            if p in kwargs:
                idx += (axis.get_nearest_index(kwargs[p]),)
            else:
                idx += (slice(None),)

        return idx

    def get_nearby_indexes(self, **kwargs):
        """
        Returns the indices bracketing the values specified. Used for linear interpolation.

        Returns:
            [type]: [description]
        """

        idx1 = ()
        idx2 = ()

        for i, p, ax in self.enumerate_axes():
            if p not in kwargs:
                idx1 += (slice(None),)
                idx2 += (slice(None))
            else:
                if kwargs[p] < ax.values[0] or ax.values[-1] < kwargs[p]:
                    # value is outside the grid
                    return None
                
                # right = True  -> bins[i - 1] < x <= bins[i]
                # right = False -> bins[i - 1] <= x < bins[i]
                i2 = np.digitize(kwargs[p], ax.values, right=True)

                # TODO: needs tolerance?
                if ax.values[i2] == kwargs[p]:
                    # i1 is right on the grid we don't interpolate
                    i1 = i2
                else:
                    i1 = i2 - 1
                
                idx1 += (i1,)
                idx2 += (i2,)

        return idx1, idx2

    def get_shell_indexes(self, size, idx=None, exclude_center=True, **kwargs):
        """
        Returns the indices surrounding a gridpoint.
        """

        if idx is None:
            idx = self.get_nearest_index(**kwargs)

        shifts = np.arange(-size, size + 1)
        ii = np.array(tuple(itertools.product(*([shifts] * len(idx)))))
        ii = np.array(idx) + ii

        if exclude_center:
            ii = np.delete(ii, (ii.shape[0] // 2), axis=0)

        # Exclude values that are outside the grid
        for i, k, ax in self.enumerate_axes():
            mask = (0 <= ii[:, i]) & (ii[:, i] < ax.values.size)
            ii = ii[mask]

        return ii

    def has_value_index(self, name):
        if self.preload_arrays or self.mmap_arrays:
            return self.value_indexes is not None and \
                   name in self.value_indexes and self.value_indexes[name] is not None
        else:
            return name in self.value_indexes and self.has_item(self.get_index_path(name))

    def get_value_index(self, name, s=None):
        if self.has_value_index(name):
            index = self.value_indexes[name]
            return index[s or ()]
        else:
            return None

    def get_value_index_unsliced(self, name, s=None):
        # Return a boolean index that is limited by the axis bound overrides.
        # The shape will be the same as the original array. Use this index to
        # load a limited subset of the data directly from the disk.
        if s is not None:
            index = np.full(self.value_indexes[name].shape, False)
            index[s] = self.value_indexes[name][s]
            return index
        else:
            return self.value_indexes[name]

    def get_mask_unsliced(self):
        shape = self.get_shape()
        if self.slice is not None:
            mask = np.full(shape, False)
            mask[self.slice] = True
        else:
            mask = np.full(shape, True)

        return mask

    def get_chunk_shape(self, name, shape, s=None):
        if self.config is not None:
            self.config.get_chunk_shape(self, name, shape, s=s)
        else:
            super(ArrayGrid, self).get_chunk_shape(name, shape, s=s)

    def has_value(self, name):
        if self.preload_arrays or self.mmap_arrays:
            return name in self.values and self.values[name] is not None and \
                   name in self.value_indexes and self.value_indexes[name] is not None
        else:
            return name in self.values and self.has_item(self.get_value_path(name)) and \
                   name in self.value_indexes and self.value_indexes[name] is not None

    def has_error(self, name):
        return False

    def has_value_at(self, name, idx, mode='any'):
        # Returns true if the specified grid point or points are filled, i.e. the
        # corresponding index values are all set to True
        if self.has_value_index(name):
            if mode == 'any':
                return np.any(self.value_indexes[name][tuple(idx)])
            elif mode == 'all':
                return np.all(self.value_indexes[name][tuple(idx)])
            else:
                raise NotImplementedError()
        else:
            return True

    def set_values(self, values, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        self.set_values_at(idx, values, s)

    def set_values_at(self, idx, values, s=None):
        for name in values:
            self.set_value_at(name, idx, values[name], s)

    def set_value(self, name, value, valid=None, s=None, **kwargs):
        idx = self.get_index(**kwargs)
        self.set_value_at(name, idx, value=value, valid=valid, s=s)

    def set_value_at(self, name, idx, value, valid=None, s=None):
        self.ensure_lazy_load()
        
        idx = Grid.rectify_index(idx)
        if self.has_value_index(name):
            if valid is None:
                valid = self.is_value_valid(name, value)
            if self.preload_arrays or self.mmap_arrays:
                self.value_indexes[name][idx] = valid
            else:
                self.save_item(self.get_index_path(name), np.array(valid), s=idx)

        idx = Grid.rectify_index(idx, s)
        if self.preload_arrays or self.mmap_arrays:
            self.values[name][idx] = value
        else:
            self.save_item(self.get_value_path(name), value, idx)

    def get_values(self, s=None, names=None, **kwargs):
        idx = self.get_index(**kwargs)
        return self.get_values_at(idx, s, names=names)

    def get_nearest_values(self, s=None, names=None, **kwargs):
        idx = self.get_nearest_index(**kwargs)
        return self.get_values_at(idx, s, names=names)

    def get_values_at(self, idx, s=None, names=None, raw=False, post_process=None, cache_key=()):
        names = names or self.values.keys()
        return {name: self.get_value_at(name, idx, s=s, raw=raw, post_process=post_process, cache_key=cache_key) for name in names}

    def get_value(self, name, s=None, squeeze=False, **kwargs):
        idx = self.get_index(**kwargs)
        v = self.get_value_at(name, idx, s=s)
        if squeeze:
            v = np.squeeze(v)
        return v

    def get_error(self, name, s=None, squeeze=False, **kwargs):
        """Return the error associated to the value `name`."""

        idx = self.get_index(**kwargs)
        v = self.get_error_at(name, idx, s)
        if squeeze:
            v = np.squeeze(v)
        return v

    def get_nearest_value(self, name, s=None, **kwargs):
        idx = self.get_nearest_index(**kwargs)
        return self.get_value_at(name, idx, s=s)

    def get_value_at(self, name, idx, s=None, raw=None, post_process=None, cache_key_prefix=()):
        # TODO: consider adding a squeeze=False option to keep exactly indexed dimensions       

        if self.value_cache is not None:
            cache_key = cache_key_prefix + (name, idx, s, raw)
            if self.value_cache.is_cached(cache_key):
                return self.value_cache.get(cache_key)

        idx = Grid.rectify_index(idx)
        
        if self.has_value_at(name, idx):
            idx = Grid.rectify_index(idx, s)
            if self.preload_arrays or self.mmap_arrays:
                v = np.array(self.values[name][idx], copy=True)
            else:
                self.ensure_lazy_load()
                v = self.load_item(self.get_value_path(name), np.ndarray, idx)
        else:
            v = None

        if post_process is not None and v is not None:
            v = post_process(v)
        
        if self.value_cache is not None:
            self.value_cache.push(cache_key, v)

        return v

    def get_error_at(self, name, idx, s=None, raw=None):
        """Return the error associated to the value `name`."""

        # Pure array grids do not currently support error.
        raise NotImplementedError()

    def load(self, filename, s=None, format=None):
        s = s or self.slice
        super().load(filename, s=s, format=format)

    def enum_values(self):
        # Figure out values from the file, if possible
        values = list(self.enum_items('/'.join([self.PREFIX_GRID, self.PREFIX_ARRAYS])))

        if len(values) == 0:
            values = list(self.values.keys())

        for k in values:
            yield k

    def save_values(self):
        for name in self.enum_values():
            if self.values[name] is not None:
                if self.preload_arrays:
                    self.logger.debug('Saving grid "{}" of size {}'.format(name, self.values[name].shape))
                    self.save_item(self.get_value_path(name), self.values[name])
                    self.logger.debug('Saved grid "{}" of size {}'.format(name, self.values[name].shape))
                elif self.mmap_arrays:
                    raise NotImplementedError()
                else:
                    shape = self.get_value_shape(name)
                    self.logger.debug('Allocating grid "{}" with size {}...'.format(name, shape))
                    self.allocate_item(self.get_value_path(name), shape, float)
                    self.logger.debug('Allocated grid "{}" with size {}. Will write directly to storage.'.format(name, shape))

    def load_values(self, s=None):
        grid_shape = self.get_shape()

        for name in self.enum_values():
            # If not running in memory saver mode, load entire array
            if self.preload_arrays:
                if s is not None:
                    self.logger.debug('Loading grid "{}" of size {}'.format(name, s))
                    self.values[name][s] = self.load_item(self.get_value_path(name), np.ndarray, s=s)
                else:
                    self.logger.debug('Loading grid "{}"'.format(name))
                    self.values[name] = self.load_item(self.get_value_path(name), np.ndarray)
                    
                if self.values[name] is not None:
                    self.value_shapes[name] = self.values[name].shape[len(grid_shape):]
                    self.logger.debug('Loaded grid "{}" of size {}'.format(name, self.value_shapes[name]))
            elif self.mmap_arrays:
                # When mmap'ing, we simply ignore the slice
                self.logger.debug('Memory mapping grid "{}"'.format(name))
                self.values[name] = self.load_item(self.get_value_path(name), np.ndarray, mmap=True)
                if self.values[name] is not None:
                    self.value_shapes[name] = self.values[name].shape[len(grid_shape):]
                    self.logger.debug('Memory mapped grid "{}" of size {}'.format(name, self.value_shapes[name]))
            else:
                # When lazy-loading, we simply ignore the slice
                shape = self.get_item_shape(self.get_value_path(name))
                self.values[name] = None
                if shape is not None:
                    self.value_shapes[name] = shape[len(grid_shape):]
                else:
                    self.value_shapes[name] = None
                
                self.logger.debug('Skipped loading grid "{}". Will read directly from storage.'.format(name))

    def save_value_indexes(self):
        for name in self.values:
            self.save_item(self.get_index_path(name), self.value_indexes[name])

    def load_value_indexes(self):
        for name in self.enum_values():
            self.value_indexes[name] = self.load_item(self.get_index_path(name), np.ndarray, s=None)

    def save_items(self):
        super().save_items()
        self.save_value_indexes()
        self.save_values()

    def load_items(self, s=None):
        super().load_items(s=s)
        self.load_value_indexes()
        self.load_values(s)

    def interpolate_value_linear(self, name, s=None, post_process=None, cache_key_prefix=(), **kwargs):
        if len(self.axes) == 1:
            return self.interpolate_value_linear1d(name, s=s, post_process=post_process, cache_key_prefix=cache_key_prefix, **kwargs)
        else:
            return self.interpolate_value_linearNd(name, s=s, post_process=post_process, cache_key_prefix=cache_key_prefix, **kwargs)

    def interpolate_value_linear1d(self, name, s=None, post_process=None, cache_key_prefix=(), **kwargs):
        idx = self.get_nearby_indexes(**kwargs)
        if idx is None:
            return None
        else:
            idx1, idx2 = idx
            if not self.has_value_at(name, idx1) or not self.has_value_at(name, idx2):
                return None

        # Parameter values to interpolate between
        # TODO: Handle the case when ax == ab
        p = list(kwargs.keys())[0]
        x = kwargs[p]
        xa = self.axes[p].values[idx1[0]]
        xb = self.axes[p].values[idx2[0]]
        a = self.get_value_at(name, idx1, s=s, post_process=post_process, cache_key_prefix=cache_key_prefix)
        b = self.get_value_at(name, idx2, s=s, post_process=post_process, cache_key_prefix=cache_key_prefix)
        m = (b - a) / (xb - xa)
        value = a + (x - xa) * m
        return value, kwargs

    def get_nearby_value(self, name, **kwargs):
        # TODO: implement multi-value version

        # Find lower and upper neighboring indices around the coordinates
        idx = self.get_nearby_indexes(**kwargs)
        if idx is None:
            return None
        else:
            idx1, idx2 = idx

        # TODO: This verification is not enough, all
        #       indices have to be checked in case of holes
        if not self.has_value_at(name, idx1) or not self.has_value_at(name, idx2):
            return None
            
        return self.get_nearby_value_at(name, idx1, idx2)
    
    def get_nearby_value_at(self, name, idx1, idx2, s=None, raw=None, post_process=None, cache_key_prefix=()):
        # TODO: implement multi-value version

        if self.value_cache is not None:
            # TODO: do we want the post process function in the cache key?
            cache_key = cache_key_prefix + (name, idx1, idx2, s, raw, post_process)
            if self.value_cache.is_cached(cache_key):
                logging.trace(f'Nearby values between {idx1} and {idx2} are found in cache.')
                return self.value_cache.get(cache_key)
            
        logging.trace(f'Loading nearby grid values between {idx1} and {idx2}.')

        # Dimensions
        D = len(idx1)

        # Generate the indices of all neighboring gridpoints
        # The shape of ii and kk is (D, 2**D)
        # TODO: what to do with squeezed indices?
        ii = np.array(list(itertools.product(*([[0, 1],] * D))))
        kk = np.array(list(itertools.product(*[[idx1[i], idx2[i]] for i in range(len(idx1))])))
        
        # Retrieve the value arrays for each of the surrounding grid points
        v = None
        for i in range(ii.shape[0]):
            y = self.get_value_at(name, kk[i], s=s, raw=raw, post_process=post_process, cache_key_prefix=cache_key_prefix)

            # Missing value at gridpoint. This means we cannot interpolate.
            # Not just return None but also cache it.
            if y is None:
                v = None
                break

            # TODO: Does this squeeze solves the problem above?
            y = np.squeeze(y)

            if v is None:
                shape = D * (2, ) + y.shape
                v = np.empty(shape)
            v[tuple(ii[i])] = y

        if self.value_cache is not None:
            self.value_cache.push(cache_key, v)
            logging.trace(f'Nearby values between {idx1} and {idx2} are written to the cache.')

        return v

    def interpolate_value_linearNd(self, name, s=None, post_process=None, cache_key_prefix=(), **kwargs):
        # TODO: implement multi-value version

        # Find lower and upper neighboring indices around the coordinates
        idx = self.get_nearby_indexes(**kwargs)
        if idx is None:
            return None
        else:
            idx1, idx2 = idx

        # Only keep dimensions where interpolation is necessary, i.e. skip
        # where axis has a single value only

        self.logger.trace('Finding values to interpolate to {} using linear Nd.'
                          .format({ k: kwargs[k] for _, k, v in self.enumerate_axes(squeeze=True) }))

        d = len(idx1)
        
        x = []
        xx = []
        for i, p, axis in self.enumerate_axes(squeeze=True): 
            x.append(kwargs[p])
            xx.append((axis.values[idx1[i]], axis.values[idx2[i]]))
        x = np.array(x)
        xx = np.array(xx)

        v = self.get_nearby_value_at(name, idx1, idx2, s=s, post_process=post_process, cache_key_prefix=cache_key_prefix)

        # Some of the nearby models are missing, interpolation cannot proceed
        if v is None:
            return None
        
        self.logger.trace('Interpolating values to {} using linead Nd.'.format(kwargs))
        
        # Perform the 1d interpolations along each axis
        for d in range(d):
            x0 = xx[d, 0]
            x1 = xx[d, 1]
            
            if (x1 != x0):
                v = v[0] + (v[1] - v[0]) / (x1 - x0) * (x[d] - x0)
            else:
                v = v[0]

        return v, kwargs

    def interpolate_value_spline(self, name, free_param, s=None, post_process=None, cache_key_prefix=(), **kwargs):
        # Interpolate a value vector using cubic splines in 1D, along a single dimension.

        # TODO: implement multi-value version
        # TODO: implement caching

        self.logger.trace('Finding values to interpolate to {} using cubic splines along {}.'.format(kwargs, free_param))

        # Find the index of the free parameter
        axis_list = list( p for i, p, ax in self.enumerate_axes())
        free_param_idx = axis_list.index(free_param)

        # Find nearest model to requested parameters
        idx = list(self.get_nearest_index(**kwargs))
        if idx is None:
            self.logger.debug('No nearest model found.')
            return None
        
        # Set all params to nearest value except the one in which we interpolate
        params = {}
        for i, p, ax in self.enumerate_axes():
            if p != free_param:
                params[p] = ax.values[idx[i]]
            else:
                params[p] = kwargs[p]

        # Determine index of models, we need an entire line of models (array of arrays)
        # to perform a cubic spline interpolation
        idx[free_param_idx] = slice(None)
        idx = Grid.rectify_index(idx)

        # Find index of models that actually exists along this line
        valid_value = self.value_indexes[name][idx]
        pars = self.axes[free_param].values[valid_value]

        # If we are at the edge of the grid, it might happen that we try to
        # interpolate over zero valid parameters, in this case return None and
        # the calling code will generate another set of random parameters
        if pars.shape[0] < 2 or kwargs[free_param] < pars.min() or pars.max() < kwargs[free_param]:
            self.logger.debug('Parameters are at the edge of grid, no interpolation possible.')
            return None

        idx = Grid.rectify_index(idx, s)
        if self.preload_arrays or self.mmap_arrays:
            value = np.array(self.values[name][idx][valid_value], copy=True)
        else:
            self.ensure_lazy_load()
            value = self.load_item(self.get_value_path(name), np.ndarray, idx)
            value = value[valid_value]

        # Apply post processing to each data vector
        # value.shape: (models, wave)
        if post_process is not None and value is not None:
            value = post_process(value)

        self.logger.debug('Interpolating values to {} using cubic splines along {}.'.format(kwargs, free_param))

        # Do as many parallel cubic spline interpolations as many wavelength bins we have
        x, y = pars, value
        fn = CubicSpline(x, y)

        return fn(kwargs[free_param]), params

    @staticmethod
    def get_axis_points(axes, padding=False, squeeze=False, interpolation='ijk'):
        # Return a dictionary of the points along each axis, either by ijk index or xyz coordinates.
        # When squeeze=True, only axes with more than 1 grid point will be included.
        # The result can be used to call np.meshgrid.

        xxi = []
        for i, p, ax in Grid.enumerate_axes_impl(axes):
            xi = None
            if interpolation == 'ijk':
                if ax.values.shape[0] == 1:
                    if not squeeze:
                        xi = np.array([0.0])
                else:
                    xi = np.arange(ax.values.shape[0], dtype=np.float64)
                    if padding:
                        xi -= 1.0
            elif interpolation == 'xyz':
                if ax.values.shape[0] > 1 or not squeeze:
                    xi = ax.values
            else:
                raise NotImplementedError()

            if xi is not None:
                xxi.append(xi)
        
        return xxi

    @staticmethod
    def get_meshgrid_points(axes, padding=False, squeeze=False, interpolation='ijk', indexing='ij'):
        # Returns a mesh grid - as a dictionary indexed by the axis names - that
        # lists all grid points, either as indexes or axis values. Meshgrid can be
        # created with any indexing method supported by numpy.

        points = ArrayGrid.get_axis_points(axes, padding=padding, squeeze=squeeze, interpolation=interpolation)
        points = np.meshgrid(*points, indexing=indexing)
        points = { p: points[i] for i, p, ax in Grid.enumerate_axes_impl(axes, squeeze=squeeze) }

        return points

    @staticmethod
    def pad_axes(orig_axes, size=1):
        padded_axes = {}
        for p in orig_axes:
            # Padded axis with linear extrapolation from the original edge values
            if orig_axes[p].values.shape[0] > 1:
                paxis = np.empty(orig_axes[p].values.shape[0] + 2 * size)
                paxis[size:-size] = orig_axes[p].values
                for i in range(size - 1, -1, -1):
                    paxis[i] = paxis[i + 1] - (paxis[i + 2] - paxis[i + 1])
                    paxis[-1 - i] = paxis[-2 - i] + (paxis[-2 - i] - paxis[-3 - i])
                padded_axes[p] = GridAxis(p, paxis, order=orig_axes[p].order)
            else:
                padded_axes[p] = orig_axes[p]
        return padded_axes
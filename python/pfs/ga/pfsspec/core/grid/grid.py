import os
import logging
import numbers
import numpy as np
from collections.abc import Iterable

from ..util.array import *
from ..pfsobject import PfsObject
from .gridaxis import GridAxis

class Grid(PfsObject):
    """
    Implements a base class to provide multidimensional grid capabilities to store and interpolate
    data vectors of any kind, including model spectra.
    """

    PREFIX_GRID = 'grid'
    PREFIX_CONST = 'const'
    PREFIX_AXIS = 'axes'
    PREFIX_ARRAYS = 'arrays'

    def __init__(self, orig=None):
        super(Grid, self).__init__(orig=orig)

        if isinstance(orig, Grid):
            self.axes = orig.axes
            self.constants = orig.constants
        else:
            self.axes = {}
            self.constants = {}

            self.init_axes()
            self.init_constants()

    def get_slice(self):
        """
        Returns the slicing of the grid.
        """
        return None

    def get_shape(self, s=None, squeeze=False):
        shape = tuple(axis.values.shape[0] for i, p, axis in self.enumerate_axes(s=s, squeeze=squeeze))
        return shape

    def get_constants(self):
        return self.constants

    def set_constants(self, constants):
        self.constants = constants

    def init_axes(self):
        pass

    def init_axis(self, name, values=None, order=None):
        order = order or len(self.axes)
        self.axes[name] = GridAxis(name, values, order=order)

    def build_axis_indexes(self):
        for p in self.axes:
            self.axes[p].build_index()

    def set_axes(self, axes):
        self.axes = {}
        for k in axes:
            self.axes[k] = type(axes[k])(k, orig=axes[k])
            self.axes[k].build_index()

    def get_axis(self, key):
        return self.axes[key]

    @staticmethod
    def enumerate_axes_impl(axes, s=None, squeeze=False):
        """
        Enumerate axes in the order of `GridAxis.order`. If squeeze is set to
        True, axes with a value vector of length <= 1 are ignored. Optionally
        slice the value vectors if `s` is specified.
        """
        order = { k: axes[k].order for k in axes.keys() }
        keys = sorted(axes.keys(), key=lambda k: order[k])
        i = 0 
        for j, k in enumerate(keys):
            axis = axes[k]

            if s is None or s[j] is None:
                sliced = False
                v = axis.values
            else:
                sliced = True
                v = axis.values[s[j]]
                
                # In case the slice resulted in a scalar
                if not isinstance(v, np.ndarray):
                    v = np.array([v])
            
            if not squeeze or v.shape[0] > 1:
                if sliced:
                    axis = GridAxis(k, values=v, order=i, orig=axis)
                    axis.build_index()

                yield i, k, axis
                i += 1

    def enumerate_axes(self, s=None, squeeze=False):
        for i, k, ax in  self.enumerate_axes_impl(self.axes, s=s, squeeze=squeeze):
            yield i, k, ax
        
    def save_params(self):
        self.save_item('/'.join((self.PREFIX_GRID, 'type')), type(self).__name__)

    def load_params(self):
        t = self.load_item('/'.join((self.PREFIX_GRID, 'type')), str)
        if t != type(self).__name__:
            Exception("Trying to load grid of the wrong type.")

    def init_constants(self):
        pass

    def init_constant(self, name):
        self.constants[name] = None

    def has_constant(self, name):
        return name in self.constants;

    def get_constant(self, name):
        return self.constants[name]

    def set_constant(self, name, value):
        self.constants[name] = value

    def save_constants(self):
        for p in self.constants:
            path = '/'.join((self.PREFIX_GRID, self.PREFIX_CONST, p))
            self.save_item(path, self.constants[p])

    def load_constants(self):
        constants = {}
        for p in self.constants:
            path = '/'.join((self.PREFIX_GRID, self.PREFIX_CONST, p))
            if self.has_item(path):
                constants[p] = self.load_item(path, np.ndarray)
        self.constants = constants

    def init_values(self):
        pass

    def allocate_values(self):
        raise NotImplementedError()

    def add_args(self, parser):
        for k in self.axes:
            self.axes[k].add_args(parser)

    def init_from_args(self, args):
        # Override physical parameters grid ranges, if specified
        # TODO: extend this to sample physically meaningful models only
        for k in self.axes:
            self.axes[k].init_from_args(args)

    def save_axes(self):
        # Make sure to write out correct value range when grid is sliced.
        slice = self.get_slice()
        for i, p, axis in self.enumerate_axes(s=slice, squeeze=False):
            path = '/'.join((self.PREFIX_GRID, self.PREFIX_AXIS, str(i), p))
            self.save_item(path, axis.values)

    def load_axes(self):
        # Old scheme: axes order is not available in the file
        # New scheme: axes names are prefixed with an order number
        
        path = '/'.join((self.PREFIX_GRID, self.PREFIX_AXIS, str(0)))
        if self.has_item(path):
            self.axes = self.load_axes_new()
        else:
            self.axes = self.load_axes_old()
        
        self.build_axis_indexes()

    def load_axes_new(self):
        axes = {}

        for i, p, axis in self.enumerate_axes():
            path = '/'.join((self.PREFIX_GRID, self.PREFIX_AXIS, str(i), p))
            if self.has_item(path):
                self.axes[p].order = i
                self.axes[p].values = self.load_item(path, np.ndarray)
                axes[p] = self.axes[p]

        return axes

    def load_axes_old(self):
        # We must keep the ordering of the axes so go with the existing
        # axis list instead of reading from the file which doesn't store
        # axes in order.

        axes = {}

        for p in self.axes:
            path = '/'.join((self.PREFIX_GRID, self.PREFIX_AXIS, p))
            if self.has_item(path):
                self.axes[p].values = self.load_item(path, np.ndarray)
                axes[p] = self.axes[p]

        return axes

    def save_items(self):
        self.save_params()
        self.save_axes()
        self.save_constants()

    def load(self, filename, s=None, format=None):
        super(Grid, self).load(filename, s=s, format=format)
        self.build_axis_indexes()

    def load_items(self, s=None):
        self.load_params()
        self.load_axes()
        self.load_constants()

#region Indexing utility functions

    @staticmethod
    def rectify_index(idx, s=None):
        idx = tuple(idx)
        if isinstance(s, Iterable):
            idx = idx + tuple(s)
        elif s is not None:
            idx = idx + (s,)

        return tuple(idx)

    def set_object_params(self, obj, idx=None, **kwargs):
        if kwargs is not None:
            self.set_object_params_kwargs(obj, **kwargs)

        if idx is not None:
            self.set_object_params_idx(obj, idx)
    
    def set_object_params_kwargs(self, obj, **kwargs):
        for p in kwargs:
            if p in self.axes and hasattr(obj, p):
                if isinstance(kwargs[p], numbers.Number):
                    setattr(obj, p, float(kwargs[p]))
                else:
                    setattr(obj, p, kwargs[p])

    def set_object_params_idx(self, obj, idx):
        for i, p, axis in self.enumerate_axes():
            setattr(obj, p, float(axis.values[idx[i]]))

#endregion
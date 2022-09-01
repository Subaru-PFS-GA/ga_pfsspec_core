import numpy as np
import mmap

from .mmapinfo import mmapinfo

class mmaparray(np.ndarray):

    # Initialization calls in order:
    #
    # - when instancing new array via constructor:
    #   1. __new__ 
    #   2. __array_finalize__ when the view is created, obj is np.ndarray
    #
    # - when calling functions on the mmaparray instance:
    #   1. __array_finalize__

    def __new__(subtype, mi):

        if not isinstance(mi, mmapinfo):
            raise ValueError("Parameter `mi` must be of type `mmapinfo`.")

        cc = 1
        for c in mi.shape:
            cc *= c

        # proto = super().__new__(subtype, mi.shape,
        #                        dtype=mi.dtype, strides=strides, order=order)

        a = np.frombuffer(mi.open_or_get(), offset=mi.offset, count=cc, dtype=mi.dtype)

        a = a.reshape(mi.shape)

        # "Cast" to mmaparray
        # This will call __array_finalize__
        a = a.view(mmaparray)

        # Create a sliced view
        if mi.slice is not None:
            a = a[mi.slice]
    
        # Set mmapinfo here, this will set the field only when a new mmaparray is
        # created via the constructor and never else
        a.mmapinfo = mi

        return a

    def __array_finalize__(self, obj):
        # - obj is np.ndarray -> creating a view on an ndarray (from __new__)
        # - obj is mmaparray -> calling a function on an existing mmaparray

        # From new object (does it happen? when?)
        if obj is None:
            return
        
        # When calling a function on an existing mmaparray, attributes
        # have to be copied since the function itself will create an np.ndarray
        # We intenationnaly leave mmapinfo None here to prevent pickling any
        # view of the array
        if isinstance(obj, mmaparray):
            pass
            self.mmapinfo = None

    # def __array_function__(self, func, types, args, kwargs):
    #     # This is called before a numpy ufunc is called on an mmaparray
    #     return super().__array_function__(func, types, args, kwargs)

    # def __array_wrap__(self, out_array, context=None):
    #     # This is called after a numpy function is called on an mmaparray
    #     return super().__array_wrap__(out_array, context)

    def __reduce__(self):
        # State is a tuple of (_reconstruct, args, data) where
        # - _reconstruct is a built-in function
        # - args is (class, ?, ?)
        # - data is (?, shape, dtype, ?, data_bytes)
        # state = super().__reduce__()

        if self.mmapinfo is None:
            raise Exception("Only the original instance of a mmaparray can be pickled. This is likely a view.")

        state = (self.__class__, (self.mmapinfo, ), {})

        return state

    # NOTE: not used, it would replace __reduce__
    # def __reduce_ex__(self, protocol):
    #     res = super().__reduce_ex__(protocol)
    #     return res

    # NOTE: not used, replaced by __reduce__
    # def __getstate__(self):
    #     attributes = self.__dict__.copy()
    #     del attributes['c']
    #     return attributes

    def __setstate__(self, state):
        self.__dict__.update(state)
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip, pickle
import h5py
import json
import multiprocessing as mp
import numbers
from collections.abc import Iterable

from .setup_logger import logger
from .constants import Constants
from pfs.ga.pfsspec.core.util.args import get_arg, is_arg
from pfs.ga.pfsspec.core.util.mmaparray import mmapinfo, mmaparray

class PfsObject():
    """
    Implements basic functions to serialize/deserialize object and save/load
    array data and pandas data frames.
    """

    def __setstate__(self, state):
        self.__dict__ = state

        # When a child process is starting use multiprocessing logger
        # This might be needed but right now getLogger should return the
        # proper logger object, even when multiprocessing
        #
        # if multiprocessing.current_process()._inheriting:
        #    logger = multiprocessing.get_logger()

    def __init__(self, orig=None):
        self.jsonomit = set([
            'jsonomit',
            'file',
            'filename',
            'fileformat',
            'filedata,'
            'hdf5file'
        ])

        if isinstance(orig, PfsObject):
            self.random_state = orig.random_state
            self.random_seed = orig.random_seed

            self.file = None

            self.filename = orig.filename
            self.fileformat = orig.fileformat
            self.filedata = orig.filedata
        else:
            self.random_state = None
            self.random_seed = None

            self.file = None

            self.filename = None
            self.fileformat = None
            self.filedata = None

    def copy(self):
        return type(self)(orig=self)
    
    def init_random_state(self, random_state=None, random_seed=None, worker_id=None):
        """
        Initialize the random state or just reseeds the existing one. When worker_id is
        specified, it is added to the seed.
        """

        random_state = random_state if random_state is not None else self.random_state        
        random_seed = random_seed if random_seed is not None else self.random_seed

        # TODO: Right now, always use the default
        if random_state is None:
            random_state = np.random
               
        # If worker ID
        if worker_id is not None:
            if random_seed is not None:
                seed = (random_seed + worker_id) % 0xFFFFFFFF
            else:
                # This would not works, seed from the outside
                raise ValueError("Value of `random_seed` must be set along with `worker_id`.")
                seed = (random_state.randint(0, np.iinfo(np.int_).max) + worker_id) % 0xFFFFFFFF
        else:
            seed = random_seed

        # Create random state with defaults if does not already exist, otherwise
        # just reseed
        if random_state is None:
            # TODO: This is causing problems with multiprocessing
            raise NotImplementedError()
            random_state = np.random.RandomState(seed=seed)
            logger.debug("Initialized random state on pid {} with seed {}".format(os.getpid(), seed))
        elif seed is not None:
            random_state.seed(seed)
            logger.debug("Re-seeded random state on pid {} with seed {}".format(os.getpid(), seed))

        # TODO: These are not saved now because they're causing problems
        #       with multiprocessing
        # self.random_state = random_state
        # self.random_seed = random_seed

        pass

    def get_random_state(self, random_state=None):
        if random_state is not None:
            return random_state
        elif self.random_state is not None:
            return self.random_state
        else:
            return np.random

    def get_arg(self, name, old_value, args=None):
        """
        Reads a command-line argument if specified, otherwise uses a default value.

        :param name: Name of the argument.
        :param old_value: Default value of the argument in case in doesn't exist in args.
        :param args: Argument dictionary. If None, self.args will be used.
        :return: The parsed command-line argument.
        """

        if hasattr(self, 'args'):
            args = args or self.args or {}
        else:
            args = args or {}

        return get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        """
        Checks if an (optional) command-line argument is specified.

        :param name: Name of the argument.
        :param args: Argument dictionary. If None, self.args will be used.
        :return: True if the optional command-line argument is specified.
        """

        if hasattr(self, 'args'):
            args = args or self.args or {}
        else:
            args = args or {}
            
        return is_arg(name, args)

    @staticmethod
    def get_format(filename):
        """
        Returns a format string based on a file extension.

        :param filename: Name of the file including the extension.
        :return: Format string
        """

        fn, ext = os.path.splitext(filename)
        if ext == '.h5':
            return 'h5'
        elif ext == '.npz':
            return 'npz'
        elif ext == '.gz':
            fn, ext = os.path.splitext(fn)
            if ext == '.npy':
                return 'numpy'
            elif ext == '.dat':
                return 'pickle'
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    @staticmethod
    def get_extension(format):
        """
        Returns the default extension based on the format string.

        :param format: Format string
        :return: Default extension.
        """

        if format == 'h5':
            return '.h5'
        elif format == 'npz':
            return '.npz'
        elif format == 'numpy':
            return 'npy.gz'
        elif format == 'pickle':
            return 'dat.gz'
        else:
            raise NotImplementedError()

    def save_json(self, filename):
        """
        Serialize the object into a JSON file.

        :param filename: Name of file.
        """

        d = self.__dict__.copy()
        for k in self.jsonomit:
            if k in d:
                del d[k]
        with open(filename, 'w') as f:
            f.write(json.dumps(d, default=self.save_json_default, indent=4))

    def save_json_default(self, o):
        return None

    def load_json(self, filename):
        """
        Loads the object from a JSON file.

        :param filename: Name of file.
        """

        with open(filename, 'r') as f:
            d = json.loads(f.read())
            for k in d:
                if k not in self.jsonomit and k in self.__dict__:
                    self.__dict__[k] = d[k]

    def save(self, filename, format='pickle', save_items_func=None):
        """
        Serializes the data items of the object in the specified format.

        :param filename: Name of the target file.
        :param format: Serialization format.
        :param save_items_func: Function that calls save_item for each data item.
        """

        logger.info("Saving {} to file {}...".format(type(self).__name__, filename))

        save_items_func = save_items_func or self.save_items

        self.filename = filename
        self.fileformat = format

        if self.fileformat in ['numpy', 'pickle']:
            with gzip.open(self.filename, 'wb') as f:
                self.file = f
                save_items_func()
                self.file = None
        elif self.fileformat == 'npz':
            self.filedata = {}
            save_items_func()
            np.savez(filename, **self.filedata)
            self.filedata = None
        elif self.fileformat == 'h5':
            save_items_func()
        else:
            raise NotImplementedError()

        logger.info("Saved {} to file {}.".format(type(self).__name__, filename))

    def save_items(self):
        """
        When implemented in derived classes, saves each data item.
        """

        raise NotImplementedError()

    def allocate_item(self, name, shape, dtype=float):
        """
        Allocates the storage space for a data item. Preallocation is supported
        in HDF5 only.

        :param name: Name or path of the data item.
        :param shape: Array shape.
        :param dtype: Array base type.
        """

        if self.fileformat != 'h5':
            raise NotImplementedError()

        if self.fileformat == 'h5':
            with h5py.File(self.filename, 'a') as f:
                if name not in f.keys():
                    chunks = self.get_chunk_shape(name, shape, None)
                    return f.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)

    def save_item(self, name, item, s=None, min_string_length=None):
        """
        Saves a data item which can be a simple type, array or pandas data frame.

        :param name: Name or path of the data item.
        :param item: The data item.
        :param s: Optional target slice. Only supported with HDF5.
        :param min_string_length: Dictionary of minimum length of pandas string columns.
        """

        logger.debug('Saving item {} with type {}'.format(name, type(item).__name__))

        if self.fileformat != 'h5' and s is not None:
            raise NotImplementedError()

        if self.fileformat == 'numpy':
            np.save(self.file, item, allow_pickle=True)
        elif self.fileformat == 'pickle':
            pickle.dump(item, self.file, protocol=4)
        elif self.fileformat == 'npz':
            self.filedata[name] = item
        elif self.fileformat == 'h5':
            self.save_item_hdf5(name, item, s=s, min_string_length=min_string_length)
        else:
            raise NotImplementedError()

    def save_item_hdf5(self, path, item, s=None, min_string_length=None):
        """
        Saves an item into an HDF5 file. Supports partial updates.

        :param path: Name or path of the dataset.
        :param item: Data item.
        :param s: Optional target slice, defaults to None.
        :param min_string_length: Dictionary of minimum length of pandas string columns.
        """

        def open_hdf5():
            if self.file is None:
                return h5py.File(self.filename, 'a')
            else:
                return self.file

        def close_hdf5(f):
            if self.file is None:
                f.close()
            else:
                pass

        f = open_hdf5()
        g, name = self.get_hdf5_group(f, path, create=True)

        if item is None:
            # Do not save if value is None
            pass
        elif isinstance(item, pd.DataFrame):
            # This is a little bit different for DataFrames when s is defined. When slicing,
            # arrays are update in place, an operation not supported by HDFStore
            # TODO: this issue could be solved by first removing the matching rows, then
            #       appending them.

            if s is None:
                if name in g:
                    del g[name]
                
                # Make sure to close the file first!
                close_hdf5(f)

                item.to_hdf(self.filename, path, mode='a', format='table', append=False, min_itemsize=min_string_length)
            else:
                # TODO: this throws an error
                # TODO: verify if s is at the end of table
                item.to_hdf(self.filename, path, mode='a', format='table', append=True, min_itemsize=min_string_length)
        elif isinstance(item, np.ndarray):
            if s is not None:
                # in-place update
                g[name][s] = item
            else:
                if name in g.keys():
                    del g[name]
                chunks = self.get_chunk_shape(name, item.shape, s=s)
                if chunks is not None:
                    g.create_dataset(name, data=item, chunks=chunks)
                    logger.debug('Saving item {} with chunks {}'.format(name, chunks))
                else:
                    g.create_dataset(name, data=item)
        elif isinstance(item, numbers.Number) or \
             isinstance(item, bool) or \
             isinstance(item, str):
            g.attrs[name] = item
        else:
            raise NotImplementedError('Unsupported type: {}'.format(type(item).__name__))

        close_hdf5(f)

    def get_chunk_shape(self, name, shape, s=None):
        """
        Calculates the optimal chunk size for an array. It is often overriden in
        derived classes to support optimized storage of various data objects.

        :param name: Name of the data item. Optimal chunking might depend on it.
        :param shape: Full shape of the data time.
        :param s: Target slice.
        """

        needchunk = False
        size = 1
        for s in shape:
            needchunk |= (s > 0x80)
            size *= s

        # TODO: Review this logic here. For some reason, tiny dimensions are chunked
        #       While longer arrays aren't. This function is definitely called during
        #       model grid import when chunking is important.

        if needchunk and size > 0x100000:
            chunks = list(shape)
            for i in range(len(chunks)):
                if chunks[i] <= 0x80:  # 128
                    chunks[i] = 1
                elif chunks[i] > 0x10000:  # 64k
                    chunks[i] = 0x1000     # 4k 
                elif chunks[i] > 0x400:  # 1k
                    pass
                else:
                    chunks[i] = 64
            return tuple(chunks)
        else:
            return None

    def load(self, filename, s=None, format=None, load_items_func=None):
        """
        Deserializes the data items of the object from the specified format.

        :param filename: Name of the source file.
        :param s: Source slice.
        :param format: Serialization format.
        :param load_items_func: Function that calls load_item for each data item.
        """

        logger.info("Loading {} from file {} with slices {}...".format(type(self).__name__, filename, s))

        load_items_func = load_items_func or self.load_items
        format = format or PfsObject.get_format(filename)

        self.filename = filename
        self.fileformat = format

        if self.fileformat in ['numpy', 'pickle']:
            with gzip.open(self.filename, 'rb') as f:
                self.file = f
                self.load_items(s=s)
                self.file = None
        elif self.fileformat == 'npz':
            self.filedata = np.load(self.filename, allow_pickle=True)
            logger.debug('Found items: {}'.format([k for k in self.filedata]))
            self.load_items(s=s)
            self.filedata = None
        elif self.fileformat == 'h5':
            self.load_items(s=s)
        else:
            raise NotImplementedError()

        logger.info("Loaded {} from file {}.".format(type(self).__name__, filename))

    def load_items(self, s=None):
        """
        When implemented in derived classes, saves each data item.
        """

        raise NotImplementedError()


    # TODO: move these functions into a serializer class that should be passed to all PfsObject
    #       instances instead of implementing them here
    def load_item(self, name, type, s=None, default=None, mmap=False, skip_mmap=False):
        """
        Loads a data item which can be a simple type, array or pandas data frame.

        :param name: Name or path of the data item.
        :param type: Type of the data item. Numpy ndarray, scalars, str and DataFrame are supported.
        :param s: Optional source slice.
        """

        # logger.debug('Loading item {} with type {} and slices {}'.format(name, type.__name__, s))

        if self.fileformat == 'numpy':
            data = np.load(self.file, allow_pickle=True)
            data = self.load_none_array(data)
            if data is not None and s is not None:
                return data[s]
            else:
                return data
        elif self.fileformat == 'pickle':
            data = pickle.load(self.file)
            if data is not None and s is not None:
                return data[s]
            else:
                return data
        elif self.fileformat == 'npz':
            if name in self.filedata:
                data = self.filedata[name]
                data = self.load_none_array(data)
                if data is not None and s is not None:
                    return data[s]
                else:
                    return data
            else:
                return None
        elif self.fileformat == 'h5':
            return self.load_item_hdf5(name, type, s=s, default=default, mmap=mmap, skip_mmap=skip_mmap)
        else:
            raise NotImplementedError()

    def get_hdf5_group(self, f, name, create=False):
        """
        Get the group and dataset name from an HDF5 path.

        :param f: An open HDF5 file object.
        :param name: Name of path of the dataset.
        :param create: If True, the data groups are created if don't exists.
        """

        # If name contains '/' split and dig down in the hierarchy.
        parts = name.split('/')
        g = f
        for part in parts[:-1]:
            # Create group if doesn't exist
            if create and part not in g:
                g = g.create_group(part)
            elif part not in g:
                return None, None
            else:
                g = g[part]
        return g, parts[-1]

    def load_item_hdf5(self, path, output_type, s=None, default=None, mmap=False, skip_mmap=False):
        """
        Loads an item from an HDF5 file. Supports partial arrays.

        :param name: Name or path of the dataset.
        :param type: Type of the data item. Numpy ndarray, scalars, str and DataFrame are supported.
        :param s: Optional source slice, defaults to None.
        :param default: Value to return if the dataset doesn't exist
        :param mmap: Load array into mmaped (shared) memory. Param `type` must be np.ndarray and no slicing is allowed.
        """

        # The supported types and operation
        # - pandas DataFrame: Read from HDF5 using pf.read_hdf, start and stop records can
        #   be specified. Throws error if dataset does not exist.
        # - numpy ndarray: Read using standard HDF5 smart indexing mechanism. Returns None
        #   if dataset does not exist.
        # - scalar float and int: Try reading it from an attribute and if not found, read
        #   using standard HDF5 smart indexing from a dataset. Returns None if neither the
        #   attributed, nor the dataset exists.
        # - str: Read from an attribute instead of a dataset. Returns None if the attribute
        #   doesn't exist.

        if output_type == pd.DataFrame:
            if s is not None:
                return pd.read_hdf(self.filename, path, start=s.start, stop=s.stop)
            else:
                return pd.read_hdf(self.filename, path)
        elif output_type == np.ndarray:
            # Try to mmap first            
            if mmap:
                with h5py.File(self.filename, 'r') as f:
                    g, name = self.get_hdf5_group(f, path, create=False)
                    if g is not None and name in g:
                        mi = mmapinfo.from_hdf5(g[name], slice=s)
                        if mi is not None:
                            return mmaparray(mi)
                        else:
                            # If skip_mmap is True, it's OK to load the array into memory
                            logger.warning(f'Cannot mmap HDF5 dataset {path} of file `{self.filename}`. Is it chunked?')
                            if not skip_mmap:
                                return None
                    else:
                        return None

            # Try without mmap
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, path, create=False)
                if g is not None and name in g:
                    # Do some smart indexing magic here because index arrays are not supported by h5py
                    # `s` can contain slices, arrays an-d integers and an ellipsis
                    # When an ndarray appears in the slice list, it will be iterated over and the output
                    # All index arrays must have the same shape
                    # is put together from pieces. This is not full fancy indexing but close enough.
                    if isinstance(s, Iterable):
                        # Look for ellipsis in s and replace it with full slices to match shape of array
                        ns = []
                        for i, ss in enumerate(s):
                            if ss is Ellipsis:
                                ns.extend([slice(None)] * (len(g[name].shape) - len(s) + 1))
                            else:
                                ns.append(ss)
                        s = tuple(ns)

                        # Calculate the output shape based on the input slice
                        output_shape = ()          # output shape
                        idx_shape = None           # shape of the index array
                        for i in range(len(s)):
                            if isinstance(s[i], (np.int32, np.int64)):
                                output_shape = output_shape + (1,)
                            elif isinstance(s[i], np.ndarray):
                                output_shape = output_shape + s[i].shape

                                # The shape of all index arrays must be the same
                                if idx_shape is not None and idx_shape != s[i].shape:
                                    raise Exception('Incompatible shapes')
                                idx_shape = s[i].shape
                            elif isinstance(s[i], slice):
                                k = len(range(*s[i].indices(g[name].shape[i])))
                                output_shape = output_shape + (k,)
                            else:
                                raise NotImplementedError()

                        if output_shape == ():
                            # If no slice is specified, that output shape is the same
                            # as the input shape
                            output_shape = g[name].shape
                        else:
                            # If the specified slice has less element than the dimensions of the
                            # HDF5 array, expand it to the full shape
                            output_shape = output_shape + g[name].shape[len(s):]

                        if idx_shape is None:
                            data = g[name][s]
                        else:
                            data = np.empty(output_shape, dtype=g[name].dtype)
                            for idx in np.ndindex(idx_shape):
                                ii = []
                                for i in range(len(s)):
                                    if isinstance(s[i], (np.int32, np.int64)):
                                        ii.append(s[i])
                                    elif isinstance(s[i], np.ndarray):
                                        ii.append(s[i][idx])
                                    elif isinstance(s[i], slice):
                                        ii.append(s[i])
                                    else:
                                        raise NotImplementedError()
                                data[idx] = g[name][tuple(ii)]
                    elif s is not None:
                        data = g[name][s]
                    else:
                        data = g[name][:]

                    # If data is a scalar, wrap it into an array.
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                else:
                    data = default

            return data
        elif output_type == float or output_type == int:
            # Try attributes first, then dataset, otherwise return None
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, path, create=False)
                if g is not None and name in g.attrs:
                    return g.attrs[name]
                elif g is not None and name in g:
                    data = g[name][()]
                    return data
                else:
                    return default
        elif output_type == bool:
            # Try attributes first, then dataset, otherwise return None
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, path, create=False)
                if g is not None and name in g.attrs:
                    return g.attrs[name]
                elif g is not None and name in g:
                    data = g[name][()]
                    return data
                else:
                    return default
        elif output_type == str:
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, path, create=False)
                if g is not None and name in g.attrs:
                    return g.attrs[name]
                else:
                    return default
        else:
            raise NotImplementedError()

    def load_none_array(self, data):
        """
        Converts empty arrays to None.

        :param data: Value to convert.
        :return: None if the array is empty, i.e. has the shape of ().
        """

        if isinstance(data, np.ndarray) and data.shape == ():
            return None
        else:
            return data

    def has_item(self, name):
        """
        Checks if a dataset exists in an HDF5 file.

        :param name: Name or path of the dataset.
        :return: True if the dataset exists.
        """

        if self.fileformat == 'h5':
            if not os.path.isfile(self.filename):
                return False
            else:
                with h5py.File(self.filename, 'r') as f:
                    g, name = self.get_hdf5_group(f, name, create=False)
                    return (g is not None) and (name in g)
        else:
            raise NotImplementedError()

    def enum_items(self, name):
        """
        Checks if a dataset exists in an HDF5 file.
        """

        if self.fileformat == 'h5':
            if not os.path.isfile(self.filename):
                return
            else:
                with h5py.File(self.filename, 'r') as f:
                    g, name = self.get_hdf5_group(f, name, create=False)
                    if name not in g:
                        return
                    else:
                        g = g[name]
                        for k in g:
                            yield k
        else:
            raise NotImplementedError()

    def get_item_shape(self, name):
        """
        Gets the shape of an HDF5 dataset. Returns None if the dataset doesn't exist.

        :param name: Name or path of the dataset.
        :return: The shape if the dataset exists, otherwise None.
        """

        if self.fileformat == 'h5':
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                if (g is None) or (name not in g):
                    return None
                else:
                    return g[name].shape
        else:
            raise NotImplementedError()

    def get_item_dtype(self, name):
        """
        Gets the dtype of an HDF5 dataset. Returns None if the dataset doesn't exist.

        :param name: Name or path of the dataset.
        :return: The dtype if the dataset exists, otherwise None.
        """
        
        if self.fileformat == 'h5':
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                if (g is None) or (name not in g):
                    return None
                else:
                    return g[name].dtype
        else:
            raise NotImplementedError()
    

    @staticmethod
    def get_dtype_default_value(dtype):
        if dtype == int:
            return 0
        elif dtype == float:
            return 0.0
        else:
            raise NotImplementedError()

    @staticmethod
    def get_dtype_invalid_value(dtype):
        if dtype == int:
            return -1
        elif dtype == float:
            return np.nan
        else:
            raise NotImplementedError()

    def plot_getax(self, ax=None, xlim=Constants.DEFAULT_PLOT_WAVE_RANGE, ylim=None):
        if ax is None:
            ax = plt.gca()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return ax

    def plot(self, ax=None, xlim=Constants.DEFAULT_PLOT_WAVE_RANGE, labels=True):
        raise NotImplementedError()
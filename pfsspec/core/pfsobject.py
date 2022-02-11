import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip, pickle
import h5py
import json
import multiprocessing
import numbers
from collections import Iterable

from .constants import Constants
from pfsspec.core.util.args import get_arg, is_arg

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
        #    self.logger = multiprocessing.get_logger()

    def __init__(self, orig=None):
        self.jsonomit = set([
            'jsonomit',
            'file',
            'filename',
            'fileformat',
            'filedata,'
            'hdf5file'
        ])

        self.logger = logging.getLogger()

        if isinstance(orig, PfsObject):
            self.file = None

            self.filename = orig.filename
            self.fileformat = orig.fileformat
            self.filedata = orig.filedata
        else:
            self.file = None

            self.filename = None
            self.fileformat = None
            self.filedata = None

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

        self.logger.info("Saving {} to file {}...".format(type(self).__name__, filename))

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

        self.logger.info("Saved {} to file {}.".format(type(self).__name__, filename))

    def save_items(self):
        """
        When implemented in derived classes, saves each data item.
        """

        raise NotImplementedError()

    def allocate_item(self, name, shape, dtype=np.float):
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

        self.logger.debug('Saving item {} with type {}'.format(name, type(item).__name__))

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

    def save_item_hdf5(self, name, item, s=None, min_string_length=None):
        """
        Saves an item into an HDF5 file. Supports partial updates.

        :param name: Name or path of the dataset.
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
        g, name = self.get_hdf5_group(f, name, create=True)

        if item is None:
            # Do not save if value is None
            pass
        elif isinstance(item, pd.DataFrame):
            # This is a little bit different for DataFrames when s is defined. When slicing,
            # arrays are update in place, an operation not supported by HDFStore
            # TODO: this issue could be solved by first removing the matching rows, then
            #       appending them.

            if s is None:
                item.to_hdf(self.filename, name, mode='a', format='table', min_itemsize=min_string_length)
            else:
                # TODO: this throws an error
                # TODO: verify if s is at the end of table
                item.to_hdf(self.filename, name, mode='a', format='table', append=True, min_itemsize=min_string_length)
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
                    self.logger.debug('Saving item {} with chunks {}'.format(name, chunks))
                else:
                    g.create_dataset(name, data=item)
        elif isinstance(item, numbers.Number):
            # TODO: now storing a single number in a separate dataset. Change this
            #       to store as an attribute. Keeping it now for compatibility.
            if name in g.keys():
                del g[name]
            g.create_dataset(name, data=item)
        elif isinstance(item, str):
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

        self.logger.info("Loading {} from file {} with slices {}...".format(type(self).__name__, filename, slice))

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
            self.logger.debug('Found items: {}'.format([k for k in self.filedata]))
            self.load_items(s=s)
            self.filedata = None
        elif self.fileformat == 'h5':
            self.load_items(s=s)
        else:
            raise NotImplementedError()

        self.logger.info("Loaded {} from file {}.".format(type(self).__name__, filename))

    def load_items(self, s=None):
        """
        When implemented in derived classes, saves each data item.
        """

        raise NotImplementedError()

    def load_item(self, name, type, s=None):
        """
        Loads a data item which can be a simple type, array or pandas data frame.

        :param name: Name or path of the data item.
        :param type: Type of the data item. Numpy ndarray, scalars, str and DataFrame are supported.
        :param s: Optional source slice.
        """

        # self.logger.debug('Loading item {} with type {} and slices {}'.format(name, type.__name__, s))

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
            return self.load_item_hdf5(name, type, s=s)
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

    def load_item_hdf5(self, name, type, s=None):
        """
        Loads an item from an HDF5 file. Supports partial arrays.

        :param name: Name or path of the dataset.
        :param type: Type of the data item. Numpy ndarray, scalars, str and DataFrame are supported.
        :param s: Optional source slice, defaults to None.
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

        if type == pd.DataFrame:
            if s is not None:
                return pd.read_hdf(self.filename, name, start=s.start, stop=s.stop)
            else:
                return pd.read_hdf(self.filename, name)
        elif type == np.ndarray:
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                if g is not None and name in g:
                    # Do some smart indexing magic here because index arrays are not supported by h5py
                    # This is not full fancy indexing!
                    shape = None
                    idxshape = None
                    if isinstance(s, Iterable):
                        for i in range(len(s)):
                            if isinstance(s[i], (np.int32, np.int64)):
                                if shape is None:
                                    shape = (1,)
                            elif isinstance(s[i], np.ndarray):
                                if shape is None or shape == (1,):
                                    shape = s[i].shape
                                if idxshape is not None and idxshape != s[i].shape:
                                    raise Exception('Incompatible shapes')
                                idxshape = s[i].shape
                            elif isinstance(s[i], slice):
                                k = len(range(*s[i].indices(g[name].shape[i])))
                                if shape is None:
                                    shape = (k, )
                                else:
                                    shape = shape + (k, )

                        if shape is None:
                            shape = g[name].shape
                        else:
                            shape = shape + g[name].shape[len(s):]

                        if idxshape is None:
                            data = g[name][s]
                        else:
                            data = np.empty(shape)
                            for idx in np.ndindex(idxshape):
                                ii = []
                                for i in range(len(s)):
                                    if isinstance(s[i], (np.int32, np.int64)):
                                        ii.append(s[i])
                                    elif isinstance(s[i], np.ndarray):
                                        ii.append(s[i][idx])
                                    elif isinstance(s[i], slice):
                                        ii.append(s[i])
                                data[idx] = g[name][tuple(ii)]
                    elif s is not None:
                        data = g[name][s]
                    else:
                        data = g[name][:]

                    # If data is a scalar, wrap it into an array.
                    if not isinstance(data, np.ndarray):
                        data = np.ndarray(data)

                    return data
                else:
                    return None
        elif type == np.float or type == np.int:
            # Try attributes first, then dataset, otherwise return None
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                if g is not None and name in g.attrs:
                    return g.attrs[name]
                elif g is not None and name in g:
                    data = g[name][()]
                    return data
                else:
                    return None
        elif type == np.bool or type == bool:
            # Try attributes first, then dataset, otherwise return None
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                if g is not None and name in g.attrs:
                    return g.attrs[name]
                elif g is not None and name in g:
                    data = g[name][()]
                    return data
                else:
                    return None
        elif type == str:
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                if g is not None and name in g.attrs:
                    return g.attrs[name]
                else:
                    return None
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
            with h5py.File(self.filename, 'r') as f:
                g, name = self.get_hdf5_group(f, name, create=False)
                return (g is not None) and (name in g)
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
        if dtype == np.int:
            return 0
        elif dtype == np.float:
            return 0.0
        else:
            raise NotImplementedError()

    @staticmethod
    def get_dtype_invalid_value(dtype):
        if dtype == np.int:
            return -1
        elif dtype == np.float:
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
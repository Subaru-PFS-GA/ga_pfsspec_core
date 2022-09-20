import numpy as np

from ..util.copy import *
from .dataset import Dataset

class ArrayDataset(Dataset):
    """
    Implements a class to store data vectors as arrays with additional parameters
    stored as a pandas DataFrame. The class supports chunked reads and writes when
    the dataset is stored in the HDF5 format.

    Since `params` can be read directly from the disk, or read and written in
    chunks, it's not always available in memory. We keep track of the number
    of records in `row_count`
    """

    POSTFIX_VALUE = 'value'

    def __init__(self, config=None, preload_arrays=None, orig=None):
        """
        Instantiates a new Dataset object.

        :param config: Configuration class inherited from 
        :param preload_arrays: When True, loads arrays (wave, flux etc.) into
            memory when the dataset is loaded. This speeds up access but can lead
            to high memory use.
        :param orig: Optionally, copy everything from an existing Dataset.
        """
        
        super(ArrayDataset, self).__init__(orig=orig)

        if isinstance(orig, Dataset):
            self.config = config if config is not None else orig.config
            self.preload_arrays = preload_arrays or orig.preload_arrays

            self.params = safe_deep_copy(orig.params)
            self.row_count = orig.row_count
            self.constants = orig.constants
            self.values = safe_deep_copy(orig.values)
            self.value_shapes = safe_deep_copy(orig.value_shapes)
            self.value_dtypes = safe_deep_copy(orig.value_dtypes)

            self.cache_chunk_id = orig.cache_chunk_id
            self.cache_chunk_size = orig.cache_chunk_size
            self.cache_dirty = orig.cache_dirty
        else:
            self.config = config            # Dataset configuration class
            self.preload_arrays = preload_arrays or False

            self.params = None              # Dataset item parameters
            self.row_count = None           # Number of rows in params
            self.constants = {}             # Global constants
            self.values = {}                # Dictionary of value arrays
            self.value_shapes = {}          # Dictionary of value array shapes
            self.value_dtypes = {}          # Dictionary of value dtypes

            self.cache_chunk_id = None
            self.cache_chunk_size = None
            self.cache_dirty = False

            self.init_values()

    def get_chunk_id(self, i, chunk_size):
        """
        Based on the index of a dataset item, it retuns the containing chunk's id.

        :param i: Index of the dataset item.
        :param chunk_size: Chunk size.
        :return: Index of the chunk.
        """

        return i // chunk_size, i % chunk_size

    def get_chunk_count(self, chunk_size):
        """
        Based on the chunk size, it returns the number of chunks. The last chunk is
        potentially partial.

        :param chunk_size: Size of the chunks.
        :return: The number of chunks.
        """

        return np.int32(np.ceil(self.get_count() / chunk_size))

    def get_chunk_slice(self, chunk_size, chunk_id):
        """
        Gets the slice selecting a particular chunk from the entire dataset.
        It applies to the params DataFrame only as the arrays are loaded partially
        when chunking is used, hence data arrays must be indexed with relative indexing.

        :param chunk_size: Size of the chunks.
        :param chunk_id: ID of the chunk. The last one will be partial.
        """

        if chunk_id is None:
            return np.s_[()]
        else:
            fr = chunk_id * chunk_size
            to = min((chunk_id + 1) * chunk_size, self.row_count)
            return np.s_[fr:to]

    def get_value_shape(self, name):
        """
        Gets the shape of a value item, i.e. the number of dataset items times
        the shape of the value array.
        """

        shape = self.get_shape() + self.value_shapes[name]
        return shape

    def init_values(self, row_count=None):
        """
        Initializes value arrays by calling `Dataset.init_value` for each value
        array defined in the DatasetConfig.
        """

        self.row_count = None
        
        if self.config is not None:
            self.config.init_values(self)

    def allocate_values(self, row_count=None):
        """
        Allocates memory or disk space for value arrays by calling 
        `Dataset.allocate_value` for each value array defined in the DatasetConfig.
        """

        if row_count is None:
            row_count = self.row_count
        else:
            self.row_count = row_count

        if self.config is not None:
            self.config.allocate_values(self, row_count)

    def init_value(self, name, shape=None, dtype=np.float, **kwargs):
        """
        Initializes a value array. When shape is not None, the array is
        allocated in memory or on the disk, depending on the storage model
        defined by preload_arrays.

        :param name: Name of the value array.
        :param shape: Shape of the value array, not including number of records.
        :param dtype: Type of the value array, defaults to `float`.
        """

        if shape is None:
            # We don't know the shape but at least we know that there is a value
            # array by the name.
            self.values[name] = None
            self.value_shapes[name] = None
            self.value_dtypes[name] = dtype
        else:
            # We know the shape, allocate the array
            if isinstance(shape, int):
                shape = (shape,)
            full_shape = (self.row_count,) + shape

            self.value_shapes[name] = shape
            self.value_dtypes[name] = dtype
          
            if self.preload_arrays:
                self.logger.info('Initializing memory for dataset value array "{}" of size {}...'.format(name, full_shape))
                self.values[name] = np.full(full_shape, self.get_dtype_invalid_value(dtype))
                self.logger.info('Initialized memory for dataset value array "{}" of size {}.'.format(name, full_shape))
            else:
                self.values[name] = None

                self.logger.info('Initializing data file for dataset value array "{}" of size {}...'.format(name, full_shape))
                if not self.has_item(self.get_value_path(name)):
                    self.allocate_item(self.get_value_path(name), full_shape, dtype=dtype)
                self.logger.info('Skipped memory initialization for dataset value array "{}". Will read random slices from storage.'.format(name))

    def allocate_value(self, name, shape=None, dtype=None, **kwargs):
        """
        Allocates the necessary memory or disk space for a value array. When shape
        or dtype are None, the values are looked up from pre-defined values so
        the value array must have already been initialized (with or without
        shape information).

        :param name: Name of the value array.
        :param shape: Shape of the value array without the number of records.
            If None, the value from `value_shapes` will be used.
        :param dtype: Type of the value array. If None, the value from `value_dtypes`
            will be used.
        """

        if name not in self.values:
            raise KeyError()

        if shape is not None:
            self.value_shapes[name] = shape
        if dtype is not None:
            self.value_dtypes[name] = dtype
        self.init_value(name, shape=self.value_shapes[name], dtype=self.value_dtypes[name])

    def get_chunk_shape(self, name, shape, s=None):
        """
        Calculates the optimal chunk size for an array. If a dataset configuration
        is used, it hooks into the config class.

        :param name: Name of the value array.
        :param shape: Full shape of the value array, including record count.
        :param s: Optional slicing, reserved, not used.
        """

        if self.config is not None:
            chunks = self.config.get_chunk_shape(self, name, shape, s=s)
        else:
            chunks = None

        if chunks is None:
            chunks = super(Dataset, self).get_chunk_shape(name, shape, s=s)

        return chunks

    #region Load and save

    def get_value_path(self, name):
        return '/'.join([self.PREFIX_DATASET, self.PREFIX_ARRAYS, name, self.POSTFIX_VALUE])

    def enum_values(self):
        # Figure out values from the file, if possible
        values = list(self.enum_items('/'.join([self.PREFIX_DATASET, self.PREFIX_ARRAYS])))

        if len(values) == 0:
            values = list(self.values.keys())

        for k in values:
            yield k

    def load_values(self, s=None):
        """
        Loads dataset value arrays from a file. When `preload_arrays` is True,
        the values are loaded into memory. When False, only metadata are loaded
        and subsequent calls to `get_value` will read directly from the file.

        :param s: Optional slice.
        """

        for name in self.enum_values():
            if self.preload_arrays:
                if s is not None:
                    self.logger.info('Loading dataset value array "{}" of size {}'.format(name, s))
                    self.values[name][s] = self.load_item(self.get_value_path(name), np.ndarray, s=s)
                    self.logger.info('Loaded dataset value array "{}" of size {}'.format(name, s))
                else:
                    self.logger.info('Loading dataset value array "{}" of size {}'.format(name, self.value_shapes[name]))
                    self.values[name] = self.load_item(self.get_value_path(name), np.ndarray)
                    self.logger.info('Loaded dataset value array "{}" of size {}'.format(name, self.value_shapes[name]))

                if self.values[name] is not None:
                    self.value_shapes[name] = self.values[name].shape[1:]
                    self.value_dtypes[name] = self.values[name].dtype
            else:
                # When lazy-loading, we simply ignore the slice
                shape = self.get_item_shape(self.get_value_path(name))
                if shape is not None:
                    self.value_shapes[name] = shape[1:]
                else:
                    self.value_shapes[name] = None
                self.value_dtypes[name] = self.get_item_dtype(self.get_value_path(name))
                
                self.logger.info('Skipped loading dataset value array "{}". Will read directly from storage.'.format(name))

    def save_values(self):
        """
        Saves dataset value arrays to a file. When `preload_arrays` is True,
        the arrays must already be allocated in memory and are written to the
        file entirely. When False, new space for the value arrays are allocated
        in the target file and subsequent calls to `set_value` will write the
        values in chunks.
        """

        for name in self.values:
            if self.values[name] is not None:
                if self.preload_arrays:
                    self.logger.info('Saving dataset value array "{}" of size {}'.format(name, self.values[name].shape))
                    self.save_item(self.get_value_path(name), self.values[name])
                    self.logger.info('Saved dataset value array "{}" of size {}'.format(name, self.values[name].shape))
                else:
                    shape = self.get_value_shape(name)
                    self.logger.info('Allocating dataset value array "{}" with size {}...'.format(name, shape))
                    self.allocate_item(self.get_value_path(name), shape, self.value_dtypes[name])
                    self.logger.info('Allocated dataset value array "{}" with size {}. Will write directly to storage.'.format(name, shape))

    #endregion
    #region Array access with chunking and caching support

    def get_value(self, name, idx=None, chunk_size=None, chunk_id=None):
        """
        Gets an indexed slice of a value array with optional chunking and data
        caching, if the array are stored on the disk, i.e. `preload_arrays` is False.

        :param name: Name of the value array.
        :param idx: Integer index array or slice into the value array. When chunking
            is used it must be relative to the chunk, otherwise absolute.
        :param chunk_size: Size of the chunks.
        :param chunk_id: ID of the chunk. The last one might be partial.
        """

        if self.preload_arrays:
            if chunk_size is not None or chunk_id is not None:
                raise NotImplementedError()
        else:
            if chunk_size is None:
                # No chunking, load directly from storage
                # Assume idx is absolute within file and not relative to chunk

                # HDF file doesn't support fancy indexing with unsorted arrays
                # so sort and reshuffle here
                if isinstance(idx, np.ndarray):
                    srt = np.argsort(idx)
                    data = self.load_item(self.get_value_path(name), np.ndarray, s=idx[srt])
                    srt = np.argsort(srt)
                    return data[srt]
                else:
                    data = self.load_item(self.get_value_path(name), np.ndarray, s=idx)
                    return data
            else:
                # Chunked lazy loading, use cache
                # Assume idx is relative to chunk
                self.read_cache(name, chunk_size, chunk_id)
        
        # Return slice from cache (or preloaded array)
        return self.values[name][idx if idx is not None else ()]

    def set_value(self, name, data, idx=None, chunk_size=None, chunk_id=None):
        """
        Sets an indexed slice of a value array with optional chunking and data
        caching, if the array are stored on the disk, i.e. `preload_arrays` is False.

        :param name: Name of the value array.
        :param data: Updates to the value array.
        :param idx: Integer index array or slice into the value array. When chunking
            is used it must be relative to the chunk, otherwise absolute.
        :param chunk_size: Size of the chunks.
        :param chunk_id: ID of the chunk. The last one might be partial.
        """

        if self.preload_arrays:
            self.values[name][idx if idx is not None else ()] = data
        else:
            if chunk_size is None:
                # No chunking, write directly to storage
                # Assume idx is absolute within file and not relative to chunk

                # HDF file doesn't support fancy indexing with unsorted arrays
                # so sort and reshuffle here
                if isinstance(idx, np.ndarray):
                    srt = np.argsort(idx)
                    self.save_item(self.get_value_path(name), data[srt], s=idx[srt])
                else:
                    self.save_item(self.get_value_path(name), data, s=idx)
            else:
                self.read_cache(name, chunk_size, chunk_id)
                self.values[name][idx if idx is not None else ()] = data
                self.cache_dirty = True

    def read_cache(self, name, chunk_size, chunk_id):
        """
        Reads a chunk from the cache, if available. Otherwise loads from the disk.
        Flushes the cache first, if dirty.

        :param name: Name of the value array.
        :param chunk_size: Size of the chunks.
        :param chunk_id: ID of the chunk. The last one might be partial.
        """

        # Flush or reset cache if necessary
        if self.cache_dirty and (self.cache_chunk_id != chunk_id or self.cache_chunk_size != chunk_size):
            self.flush_cache_all(chunk_size, chunk_id)
        elif self.cache_chunk_id is None or self.cache_chunk_id != chunk_id or self.cache_chunk_size != chunk_size:
            self.reset_cache_all(chunk_size, chunk_id)

        # Load and cache current
        data = self.values[name]
        if data is None:
            self.logger.debug('Reading dataset chunk `{}:{}` from disk.'.format(name, chunk_id))
            s = self.get_chunk_slice(chunk_size, chunk_id)
            data = self.load_item(self.get_value_path(name), np.ndarray, s=s)
            self.cache_dirty = False
        
        self.values[name] = data

    def reset_cache_all(self, chunk_size, chunk_id):
        """
        Resets the cache for all values. Sets new chunking.

        :param chunk_size: Size of the chunks.
        :param chunk_id: ID of the chunk. The last one might be partial.
        """
        for name in self.values:
            self.values[name] = None
        
        self.cache_chunk_id = chunk_id
        self.cache_chunk_size = chunk_size
        self.cache_dirty = False

    def flush_cache_item(self, name):
        """
        Flush the contents of a value array to the disk.

        :param name: Name of the value array
        """

        s = self.get_chunk_slice(self.cache_chunk_size, self.cache_chunk_id)
        data = self.values[name]
        if data is not None:
            self.save_item(self.get_value_path(name), data, s=s)
        self.values[name] = None

    def flush_cache_all(self, chunk_size, chunk_id, id_key='id'):
        """
        Flush the contents of the value array cache to the disk. Also save the
        corresponding rows from the params DataFrame, even though it's kept in
        memory. This is to ensure consistency.

        :param chunk_size: Size of the chunks.
        :param chunk_id: ID of the chunk. The last one might be partial.
        :param id_key: The dictionary item to be used as index. Defaults to "id".
        """

        self.logger.debug('Flushing dataset chunk `:{}` to disk.'.format(self.cache_chunk_id))

        for name in self.values:
            self.flush_cache_item(name)

        # TODO: move to merge
        # # Sort and save the last chunk of the parameters
        # if self.params is not None:
        #     self.params.sort_values(id_key, inplace=True)
        # msl = self.get_min_string_length()
        # s = self.get_chunk_slice(self.cache_chunk_size, self.cache_chunk_id)
        # self.save_item('params', self.params, s=s, min_string_length=msl)

        # # TODO: The issue with merging new chunks into an existing DataFrame is that
        # #       the index is rebuilt repeadately, causing an ever increasing processing
        # #       time. To prevent this, we reset the params DataFrame here but it should
        # #       only happen during building a dataset. Figure out a better solution to this.
        # self.params = None

        self.logger.debug('Flushed dataset chunk {} to disk.'.format(self.cache_chunk_id))

        self.reset_cache_all(chunk_size, chunk_id)

    #endregion
    #region Split, merge and filter

    def where(self, f):
        """
        Filters the dataset by a filter specified as a boolean array or
        pandas Series.

        :param f: Filter array or Series
        :return: Filtered dataset
        """

        ds = super().where(f)

        if f is None:
            return ds
        else:
            if self.preload_arrays:
                for name in self.values:
                    ds.values[name] = self.values[name][f]
            else:
                pass

    def merge(self, b):
        """
        Merges another dataset into the current one.
        """

        if self.preload_arrays and b.preload_arrays:
            ds = super().merge(b)

            for name in self.values:
                ds.values[name] = np.concatenate([self.value[name], b.values[name]], axis=0)

            return ds
        else:
            raise Exception('To merge, both datasets should be preloaded into memory.')

    def merge_chunk(self, b, chunk_size, chunk_id, chunk_offset):
        # We assume that chunk sizes are compatible and the destination dataset (self)
        # is properly preallocated

        for name in self.values:
            data = b.get_value(name, chunk_size=chunk_size, chunk_id=chunk_id)
            self.set_value(name, data, chunk_size=chunk_size, chunk_id=chunk_offset + chunk_id)

    #endregion
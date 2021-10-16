import numpy as np
import pandas as pd

from pfsspec.core.pfsobject import PfsObject

class Dataset(PfsObject):
    """
    Implements a class to store data vectors as arrays with additional parameters
    stored as a pandas DataFrame. The class supports chunked reads and writes when
    the dataset is stored in the HDF5 format.

    Since `params` can be read directly from the disk, or read and written in
    chunks, it's not always available in memory. We keep track of the number
    of records in `row_count`
    """

    def __init__(self, config=None, preload_arrays=None, orig=None):
        """
        Instantiates a new Dataset object.

        :param config: Configuration class inherited from 
        :param preload_arrays: When True, loads arrays (wave, flux etc.) into
            memory when the dataset is loaded. This speeds up access but can lead
            to high memory use.
        :param orig: Optionally, copy everything from an existing Dataset.
        """
        
        super(Dataset, self).__init__(orig=orig)

        if isinstance(orig, Dataset):
            self.config = config if config is not None else orig.config
            self.preload_arrays = preload_arrays or orig.preload_arrays

            self.params = orig.params.copy()
            self.row_count = orig.row_count
            self.constants = orig.constants
            self.values = orig.values
            self.value_shapes = orig.value_shapes
            self.value_dtypes = orig.value_dtypes

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

    #region Counts and chunks

    def get_count(self):
        """
        Gets the number of dataset items.

        :return: The number of dataset items.
        """

        return self.row_count

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

    def get_shape(self):
        """
        Gets the shape of the dataset, i.e. the number of dataset items.
        """

        return (self.row_count,)

    def get_value_shape(self, name):
        """
        Gets the shape of a value item, i.e. the number of dataset items times
        the shape of the value array.
        """

        shape = self.get_shape() + self.value_shapes[name]
        return shape
            
    #endregion

    def init_values(self, row_count=None):
        """
        Initializes value arrays by calling `Dataset.init_value` for each value
        array defined in the DatasetConfig.
        """

        self.row_count = None
        
        if self.config is not None:
            self.config.init_values(self)

    def allocate_values(self, row_count):
        """
        Allocates memory or disk space for value arrays by calling 
        `Dataset.allocate_value` for each value array defined in the DatasetConfig.
        """

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
                if not self.has_item(name):
                    self.allocate_item(name, full_shape, dtype=dtype)
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

    def get_min_string_length(self):
        msl = {}
        for col in self.params:
            if self.params[col].dtype == str:
                msl[col] = 50           # TODO: how to make it a param?

        return msl

    #region Load and save

    def load(self, filename, format='h5', s=None):
        """
        Loads a dataset from a file.

        :param filename: Path to the file.
        :param format: Format string, defaults to 'h5'.
        :param s: Optional slice.
        """

        super(Dataset, self).load(filename, format=format, s=s)

        self.logger.info("Loaded dataset with shapes:")
        self.logger.info("  params:  {}".format(self.params.shape))
        self.logger.info("  columns: {}".format(self.params.columns))

    def load_items(self, s=None):
        """
        Loads dataset items (parameters, constants, value arrays) from a file.

        :param s: Optional slice.
        """

        self.params = self.load_item('params', pd.DataFrame, s=s)
        self.row_count = self.params.shape[0]
        self.load_constants()
        self.load_values(s=s)

    def load_constants(self):
        """
        Loads dataset constants from a file.
        """

        constants = {}
        for p in self.constants:
            if self.has_item(p):
                constants[p] = self.load_item(p, np.ndarray)
        self.constants = constants

    def load_values(self, s=None):
        """
        Loads dataset value arrays from a file. When `preload_arrays` is True,
        the values are loaded into memory. When False, only metadata are loaded
        and subsequent calls to `get_value` will read directly from the file.

        :param s: Optional slice.
        """

        for name in self.values:
            if self.preload_arrays:
                if s is not None:
                    self.logger.info('Loading dataset value array "{}" of size {}'.format(name, s))
                    self.values[name][s] = self.load_item(name, np.ndarray, s=s)
                    self.logger.info('Loaded dataset value array "{}" of size {}'.format(name, s))
                else:
                    self.logger.info('Loading dataset value array "{}" of size {}'.format(name, self.value_shapes[name]))
                    self.values[name] = self.load_item(name, np.ndarray)
                    self.logger.info('Loaded dataset value array "{}" of size {}'.format(name, self.value_shapes[name]))
                self.value_shapes[name] = self.values[name].shape[1:]
                self.value_dtypes[name] = self.values[name].dtype
            else:
                # When lazy-loading, we simply ignore the slice
                shape = self.get_item_shape(name)
                if shape is not None:
                    self.value_shapes[name] = shape[1:]
                else:
                    self.value_shapes[name] = None
                self.value_dtypes[name] = self.get_item_dtype(name)
                
                self.logger.info('Skipped loading dataset value array "{}". Will read directly from storage.'.format(name))

    def save(self, filename, format='h5'):
        """
        Saves the dataset to a file.

        :param filename: Path to the file.
        :param format: Format string, defaults to 'h5'.
        """

        super(Dataset, self).save(filename, format=format)

        self.logger.info("Saved dataset with shapes:")
        if self.params is not None:
            self.logger.info("  params:  {}".format(self.params.shape))
            self.logger.info("  columns: {}".format(self.params.columns))

    def save_items(self):
        """
        Saves dataset items (parameters, constants, value arrays) to a file.
        """

        msl = self.get_min_string_length()
        self.save_item('params', self.params, min_string_length=msl)
        self.save_constants()
        self.save_values()

    def save_constants(self):
        """
        Saves dataset constants to a file.
        """

        for p in self.constants:
            self.save_item(p, self.constants[p])

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
                    self.save_item(name, self.values[name])
                    self.logger.info('Saved dataset value array "{}" of size {}'.format(name, self.values[name].shape))
                else:
                    shape = self.get_value_shape(name)
                    self.logger.info('Allocating dataset value array "{}" with size {}...'.format(name, shape))
                    self.allocate_item(name, shape, self.value_dtypes[name])
                    self.logger.info('Allocated dataset value array "{}" with size {}. Will write directly to storage.'.format(name, shape))

    #endregion
    #region Params access

    def get_params(self, names, idx=None, chunk_size=None, chunk_id=None):
        """
        Returns an indexed part of the parameters DataFrame. When `chunk_size` 
        and `chunk_id` are defined, `idx` is assumed to be relative to the chunk,
        otherwise absolute.

        :param names: List of names columns to return. When None, all columns are
            returned.          
        :param idx: Row indexes to be returned. When None, all rows are returned.
            When chunking, indexes must be relative to the chunk and when None,
            the entire chunk is returned.
        :param chunk_size: Chunk size.
        :param chunk_id: Chunk index, the last one might point to a partial chunk.
        """

        # Here we assume that the params DataFrame is already in memory
        if chunk_id is None:
            if names is None:
                return self.params.iloc[idx]
            else:
                return self.params[names].iloc[idx]
        else:
            s = self.get_chunk_slice(chunk_size, chunk_id)
            if names is None:
                return self.params.iloc[s].iloc[idx if idx is not None else slice(None)]
            else:
                return self.params[names].iloc[s].iloc[idx if idx is not None else slice(None)]

    def set_params(self, names, values, idx=None, chunk_size=None, chunk_id=None):
        """
        Sets the values of the parameters DataFrame based on a numpy array. 

        :param names: Names of columns to be updated.
        :param values: Value array containing the updates. Column index must be
            the last.
        :param idx: Optional integer array indexing the rows. When chunking is used,
            indexes must be relative to the chunk, otherwise absolute.
        :param chunk_size: Chunk size.
        :param chunk_id: Chunk index, the last one might point to a partial chunk.
        """
        
        if idx is None:
            idx = np.arange(0, values.shape[0], dtype=np.int)
        
        # Update the rows of the DataFrame
        for i, name in enumerate(names):
            if name not in self.params.columns:
                self.params.loc[:, name] = np.zeros((), dtype=values.dtype)
            
            if chunk_id is None:
                self.params[name].iloc[idx] = values[..., i]
            else:
                self.params[name].iloc[chunk_id * chunk_size + idx] = values[..., i]

    def append_params_row(self, row, id_key='id'):
        """
        Appends a row to the params DataFrame. If the params DataFrame is None,
        it will consists of a single row.

        :param row: A dictionary of values to be appended.
        :param id_key: The dictionary item to be used as index. Defaults to "id".
        """

        if self.params is None:
            self.params = pd.DataFrame(row, index=[row[id_key]])
        else:
            self.params = self.params.append(pd.Series(row, index=self.params.columns, name=row[id_key]))

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
                    data = self.load_item(name, np.ndarray, s=idx[srt])
                    srt = np.argsort(srt)
                    return data[srt]
                else:
                    data = self.load_item(name, np.ndarray, s=idx)
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
                    self.save_item(name, data[srt], s=idx[srt])
                else:
                    self.save_item(name, data, s=idx)
            else:
                self.read_cache(name, chunk_size, chunk_id)
                self.values[name][idx if idx is not None else ()] = data
                self.cache_dirty = True
      
    def has_item(self, name):
        if self.preload_arrays:
            return getattr(self, name) is not None
        else:
            return super(Dataset, self).has_item(name)

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
            data = self.load_item(name, np.ndarray, s=s)
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
            self.save_item(name, data, s=s)
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

    def reset_index(self, df):
        """
        Resets the index of a dataframe to be continuos and incremental.

        :param df: The DataFrame.
        """

        df.index = pd.RangeIndex(len(df.index))

    def get_split_index(self, split_value):
        """
        Returns the index where the dataset should be split given a
        split value.

        :param split_value: Split value (float between 0.0 and 1.0).
        """

        split_index = int((1 - split_value) *  self.get_count())
        return split_index

    def get_split_ranges(self, split_index):
        """
        Returns two pandas Series that mask the two parts of the dataset
        when split at `split_index`.

        :param split_index: The index where the dataset is split at.
        """

        a_range = pd.Series(split_index * [True] + (self.get_count() - split_index) * [False])
        b_range = ~a_range
        return a_range, b_range

    def split(self, split_value):
        """
        Splits the dataset into two at a given split value.

        :param split_value: Split value (float between 0.0 and 1.0).
        :return: split_index, dataset A and dataset B.
        """

        split_index = self.get_split_index(split_value)
        a_range, b_range = self.get_split_ranges(split_index)

        a = self.where(a_range)
        b = self.where(b_range)

        return split_index, a, b

    def where(self, f):
        """
        Filters the dataset by a filder specified as a boolean array or
        pandas Serier.

        :param f: Filter array or Series
        :return: Filtered dataset
        """

        if f is None:
            ds = type(self)(orig=self)
            return ds
        else:
            ds = type(self)(orig=self)
            ds.params = self.params[f]
            self.reset_index(ds.params)

            if self.preload_arrays:
                if self.constant_wave:
                    ds.wave = self.wave
                else:
                    ds.wave = self.wave[f]
                ds.flux = self.flux[f]
                ds.error = self.error[f] if self.error is not None else None
                ds.mask = self.mask[f] if self.mask is not None else None
            else:
                if self.constant_wave:
                    ds.wave = self.wave
                else:
                    ds.wave = None
                ds.flux = None
                ds.error = None
                ds.mask = None

            ds.shape = (ds.params.shape[0],) + self.shape[1:]
            
            return ds

    def merge(self, b):
        """
        Merges another dataset into the current one.
        """

        if self.preload_arrays and b.preload_arrays:
            ds = Dataset()

            ds.params = pd.concat([self.params, b.params], axis=0)
            self.reset_index(ds.params)

            if self.constant_wave:
                ds.wave = self.wave
            else:
                ds.wave = np.concatenate([self.wave, b.wave], axis=0)
            ds.flux = np.concatenate([self.flux, b.flux], axis=0)
            ds.error = np.concatenate([self.error, b.error], axis=0) if self.error is not None and b.error is not None else None
            ds.mask = np.concatenate([self.mask, b.mask], axis=0) if self.mask is not None and b.mask is not None else None
            ds.shape = (ds.params.shape[0],) + self.shape[1:]

            return ds
        else:
            raise Exception('To merge, both datasets should be preloaded into memory.')

    def merge_chunk(self, b, chunk_size, chunk_id, chunk_offset):
        # We assume that chunk sizes are compatible and the destination dataset (self)
        # is properly preallocated
        def copy_item(name):
            data = b.get_value(name, chunk_size=chunk_size, chunk_id=chunk_id)
            self.set_value(name, data, chunk_size=chunk_size, chunk_id=chunk_offset + chunk_id)
        
        if self.constant_wave:
            self.wave = b.wave
        else:
            copy_item('wave')

        for name in ['flux', 'error', 'mask']:
            copy_item(name)

        # When merging the parameters, the id field must be shifted by the offset
        self.params = b.get_params(None, chunk_size=chunk_size, chunk_id=chunk_id)
        self.params['id'] = self.params['id'].apply(lambda x: x + chunk_size * chunk_offset)
        self.params.set_index(pd.Index(list(self.params['id'])), drop=False, inplace=True)

        pass

    #endregion

    

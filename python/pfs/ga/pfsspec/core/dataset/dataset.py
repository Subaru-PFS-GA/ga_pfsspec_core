import numpy as np
import pandas as pd

from pfs.ga.pfsspec.core.pfsobject import PfsObject

class Dataset(PfsObject):

    PREFIX_DATASET = 'dataset'
    PREFIX_CONST = 'const'
    PREFIX_PARAMS = 'params'
    PREFIX_ARRAYS = 'arrays'

    def __init__(self, orig=None):
        super(Dataset, self).__init__(orig=orig)

        if isinstance(orig, Dataset):
            self.params = orig.params.copy()
            self.row_count = orig.row_count
            self.constants = orig.constants
        else:
            self.params = None              # Dataset item parameters
            self.row_count = None           # Number of rows in params
            self.constants = {}             # Global constants

            self.init_constants()

    #region Counts and chunks

    def get_count(self):
        """
        Gets the number of dataset items.

        :return: The number of dataset items.
        """

        return self.row_count

    def get_shape(self):
        """
        Gets the shape of the dataset, i.e. the number of dataset items.
        """

        return (self.row_count,)

    def get_min_string_length(self):
        msl = {}
        for col in self.params:
            if self.params[col].dtype == str:
                msl[col] = 50           # TODO: how to make it a param?

        return msl
            
    #endregion
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

        self.params = self.load_item('/'.join([self.PREFIX_DATASET, self.PREFIX_PARAMS]), pd.DataFrame, s=s)
        self.row_count = self.params.shape[0]
        self.load_constants()
        self.load_values(s=s)

    def init_constants(self):
        pass

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
        raise NotImplementedError()

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
        self.save_item('/'.join([self.PREFIX_DATASET, self.PREFIX_PARAMS]), self.params, min_string_length=msl)
        self.save_constants()
        self.save_values()

    def save_constants(self):
        """
        Saves dataset constants to a file.
        """

        for p in self.constants:
            self.save_item(p, self.constants[p])

    def save_values(self):
        raise NotImplementedError()

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
        Filters the dataset by a filter specified as a boolean array or
        pandas Series.

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
            ds.shape = (ds.params.shape[0],) + self.shape[1:]
            return ds

    def merge(self, b):
        """
        Merges another dataset into the current one.
        """

        ds = Dataset()
        ds.params = pd.concat([self.params, b.params], axis=0)
        self.reset_index(ds.params)
        ds.shape = (ds.params.shape[0],) + self.shape[1:]
        return ds

    def merge_chunk(self, b, chunk_size, chunk_id, chunk_offset):
        # We assume that chunk sizes are compatible and the destination dataset (self)
        # is properly preallocated

        # When merging the parameters, the id field must be shifted by the offset
        self.params = b.get_params(None, chunk_size=chunk_size, chunk_id=chunk_id)
        self.params['id'] = self.params['id'].apply(lambda x: x + chunk_size * chunk_offset)
        self.params.set_index(pd.Index(list(self.params['id'])), drop=False, inplace=True)

    #endregion

    

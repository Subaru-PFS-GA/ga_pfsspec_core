import mmap
import logging
import numpy as np

_mmapinfo__files = {}
_mmapinfo__mmaps = {}

class mmapinfo():

    def __init__(self, filename, offset=0, size=None, dtype=np.float64, shape=None, slice=None):
        self.__filename = filename

        self.__offset = offset
        self.__size = size
        self.__dtype = dtype
        self.__shape = shape
        self.__slice = slice

    @staticmethod
    def from_hdf5(dataset, slice=None):
        filename = dataset.file.filename
        offset = dataset.id.get_offset()
        size = dataset.id.get_storage_size()
        dtype = dataset.dtype
        shape = dataset.shape

        if offset is None:
            logging.warn('Dataset from HDF5 cannot be memmapped, offset is None meaning the array might not be contigous.')
            return None
        else:
            return mmapinfo(filename, offset=offset, size=size, dtype=dtype, shape=shape, slice=slice)

    def __get_filename(self):
        return self.__filename

    filename = property(__get_filename)

    def __get_offset(self):
        return self.__offset

    offset = property(__get_offset)

    def __get_size(self):
        return self.__size

    size = property(__get_size)

    def __get_dtype(self):
        return self.__dtype

    dtype = property(__get_dtype)

    def __get_shape(self):
        return self.__shape

    shape = property(__get_shape)

    def __get_slice(self):
        return self.__slice

    slice = property(__get_slice)

    def __get_closed(self):
        global __mmaps

        if self.__filename in __mmaps:
            return __mmaps[self.__filename].closed
        else:
            return True

    closed = property(__get_closed)

    def __getstate__(self):
        state = {}
        state.update(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def open_or_get(self):
        global __mmaps
        global __files

        if self.__filename not in __mmaps:
            f = open(self.__filename, 'br')
            mm = mmap.mmap(f.fileno(), length=0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)

            __files[self.__filename] = f
            __mmaps[self.__filename] = mm

        return __mmaps[self.__filename]

    def close(self):
        global __mmaps
        global __files

        if self.__filename in __mmaps:
            __mmaps[self.__filename].close()
            del __mmaps[self.__filename]

            __files[self.__filename].close()
            del __files[self.__filename]
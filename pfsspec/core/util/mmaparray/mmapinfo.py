import mmap
import numpy as np

_mmapinfo__files = {}
_mmapinfo__mmaps = {}

class mmapinfo():

    def __init__(self, filename, offset=0, size=None, dtype=np.float64, shape=None):
        self.__filename = filename

        self.__offset = offset
        self.__size = size
        self.__dtype = dtype
        self.__shape = shape

    @staticmethod
    def from_hdf5(dataset):
        filename = dataset.file.filename
        offset = dataset.id.get_offset()
        size = dataset.id.get_storage_size()
        dtype = dataset.dtype
        shape = dataset.shape

        return mmapinfo(filename, offset=offset, size=size, dtype=dtype, shape=shape)

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

        # return {'_mmapinfo__filename': self.__filename,
        #         '_mmapinfo__offset': self.__offset,
        #         '_mmapinfo__size': self.__size,
        #         '_mmapinfo__dtype': self.__dtype,
        #         '_mmapinfo__shape': self.__shape}

    def __setstate__(self, state):
        self.__dict__.update(state)

        # self.__dict__['_mmapinfo__filename'] = state.pop('_mmapinfo__filename', None)
        # self.__dict__['_mmapinfo__offset'] = state.pop('_mmapinfo__offset', None)
        # self.__dict__['_mmapinfo__size'] = state.pop('_mmapinfo__size', None)
        # self.__dict__['_mmapinfo__dtype'] = state.pop('_mmapinfo__dtype', None)
        # self.__dict__['_mmapinfo__shape'] = state.pop('_mmapinfo__shape', None)

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
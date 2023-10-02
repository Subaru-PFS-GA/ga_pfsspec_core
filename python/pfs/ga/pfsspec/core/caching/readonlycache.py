from datetime import datetime
from collections import namedtuple
import numpy as np
import types

CacheItem = namedtuple('CacheItem', ['timestamp', 'value'])

class ReadOnlyCache():
    # Implements a key-value read-only cache
    
    def __init__(self, max_size=100, max_age=None, orig=None):

        if not isinstance(orig, ReadOnlyCache):
            self.max_size = max_size
            self.max_age = max_age

            self.dict = {}
        else:
            self.max_size = max_size if max_size is not None else orig.max_size
            self.max_age = max_age if max_age is not None else orig.max_age

            self.dict = orig.dict

    def __len__(self):
        return len(self.dict)

    def reset(self):
        self.dict = {}

    def sanitize_key(self, key):
        if isinstance(key, tuple):
            return tuple(self.sanitize_key(i) for i in key)
        if isinstance(key, list):
            return tuple(self.sanitize_key(i) for i in key)
        elif isinstance(key, dict):
            return tuple([ self.sanitize_key((k, v)) for k, v in key.items() ])
        elif isinstance(key, np.ndarray):
            return self.sanitize_key(tuple(key))
        elif isinstance(key, slice):
            return (key.start, key.stop, key.step)
        elif isinstance(key, types.FunctionType):
            return key.__qualname__
        else:
            return key

    def is_cached(self, key):
        key = self.sanitize_key(key)
        return key in self.dict
    
    def push(self, key, value):
        key = self.sanitize_key(key)
        if key not in self.dict:
            self.dict[key] = CacheItem(datetime.now(), value)
        else:
            raise Exception('Duplicate key in cache.')

    def pop(self, key):
        key = self.sanitize_key(key)
        if key in self.dict:
            value = self.dict[key].value
            del self.dict[key]
            return value
        else:
            raise Exception('Key not in cache.')

    def get(self, key):
        key = self.sanitize_key(key)
        if key in self.dict:
            self._touch(key)
            value = self.dict[key].value
            return value
        else:
            raise Exception('Key not in cache.')
    
    def _touch(self, key):
        if key in self.dict:
            value = self.dict[key].value
            del self.dict[key]
            self.dict[key] = CacheItem(datetime.now(), value)
        else:
            raise Exception('Key not in cache.')
        
    def flush(self):
        # Find items that can be purged from the cache
        if self.max_age is not None or self.max_size is not None:
            # Sort dictionary keys ordered by the timestamp

            skeys = sorted(self.dict.keys(), key=lambda k: self.dict[k].timestamp)

            if self.max_size is not None:
                for k in skeys[self.max_size:]:
                    del self.dict[k]

            if self.max_age is not None:
                # TODO: throw away entries older than some time
                raise NotImplementedError()
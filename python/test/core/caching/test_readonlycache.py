import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.caching import ReadOnlyCache


class TestReadOnlyCache(TestBase):
    def test_flush(self):
        c = ReadOnlyCache()

        c.push(2, 2)
        c.push(1, 1)
        

        c.flush()

    def test_sanitize_key(self):
        c = ReadOnlyCache()

        c.push(np.array([2, 2]), 2)
        c.push(np.array([[2], [2]]), 2)

        c.push({ 'a': np.array([[2], [2]]) }, 2)

        c.push(slice(None), 2)
import numpy as np
import multiprocessing as mp
import pickle
import h5py
import mmap

from .testbase import testbase
from pfsspec.core.util.mmaparray import mmaparray, mmapinfo

def multiprocessing_worker(a):
    assert a.mmapinfo is not None
    assert a.shape == (154044, 9725)
    assert a.itemsize == 8
    assert a.nbytes == 11984623200

class Test_mmaparray(testbase):

    def test_create(self):
        with self.get_test_hdf5() as f:
            d = f['grid']['arrays']['flux']['pca']['eigv']
            mi = mmapinfo.from_hdf5(d)
            
        a = mmaparray(mi)

        self.assertIsInstance(a, mmaparray)
        self.assertIsNotNone(a.mmapinfo)
        self.assertEqual(a.shape, (154044, 9725))

    def test_slice(self):
        with self.get_test_hdf5() as f:
            d = f['grid']['arrays']['flux']['pca']['eigv']
            mi = mmapinfo.from_hdf5(d)
            
        a = mmaparray(mi)
        b = a[:5, :5]

        self.assertIsInstance(b, mmaparray)
        self.assertEqual(b.shape, (5, 5))
        self.assertIsNone(b.mmapinfo)

    def test_ufunc(self):
        with self.get_test_hdf5() as f:
            d = f['grid']['arrays']['flux']['pca']['eigv']
            mi = mmapinfo.from_hdf5(d)
            
        a = mmaparray(mi)
        b = np.sum(a[:5, :5])

        self.assertIsInstance(b, mmaparray)
        self.assertIsNone(b.mmapinfo)

    def test_pickle(self):
        with self.get_test_hdf5() as f:
            d = f['grid']['arrays']['flux']['pca']['eigv']
            mi = mmapinfo.from_hdf5(d)
            
        a = mmaparray(mi)
        s = pickle.dumps(a)

        b = pickle.loads(s)
        self.assertIsInstance(b, mmaparray)
        self.assertIsNotNone(b.mmapinfo)
        self.assertEqual(b.shape, (154044, 9725))

        c = a[:5, :5]
        self.assertIsInstance(c, mmaparray)
        self.assertRaises(Exception, pickle.dumps, (c,))

    def test_multiprocessing(self):
        with self.get_test_hdf5() as f:
            d = f['grid']['arrays']['flux']['pca']['eigv']
            mi = mmapinfo.from_hdf5(d)
            
        a = mmaparray(mi)

        pool = mp.Pool(1)
        pool.apply(multiprocessing_worker, (a,))

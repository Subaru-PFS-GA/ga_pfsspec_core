import numpy as np
import pickle
from unittest import TestCase

from .testbase import testbase
from pfs.ga.pfsspec.core.util.mmaparray import mmapinfo

class Test_mmapinfo(testbase):

    def create_mmapinfo(self):
        fn = self.get_test_hdf5_filename()
        mi = mmapinfo(fn, slice=np.s_[:100])
        return mi

    def test_init(self):
        mi = self.create_mmapinfo()
        self.assertEqual(mi.filename, self.get_test_hdf5_filename())

    def test_open_or_get_close(self):
        mi = self.create_mmapinfo()
        mm = mi.open_or_get()

        self.assertFalse(mm.closed)
        self.assertFalse(mi.closed)

        mi.close()

        self.assertTrue(mm.closed)
        self.assertTrue(mi.closed)

    def test_pickle(self):
        mi1 = self.create_mmapinfo()
        mi1.open_or_get()

        self.assertFalse(mi1.closed)

        s = pickle.dumps(mi1)

        mi2 = pickle.loads(s)

        self.assertFalse(mi2.closed)

        mi1.close()

        self.assertTrue(mi2.closed)

    def test_from_hdf5(self):
        with self.get_test_hdf5() as f:
            d = f['grid']['arrays']['flux']['pca']['eigv']
            mi = mmapinfo.from_hdf5(d)

    def test_picke_from_hdf5(self):
        with self.get_test_hdf5() as f:
            d = f['grid']['arrays']['flux']['pca']['eigv']
            mi1 = mmapinfo.from_hdf5(d)

        s = pickle.dumps(mi1)
        mi2 = pickle.loads(s)

        self.assertEqual(mi1.filename, mi2.filename)
        self.assertEqual(mi1.offset, mi2.offset)
        self.assertEqual(mi1.size, mi2.size)
        self.assertEqual(mi1.dtype, mi2.dtype)
        self.assertEqual(mi1.shape, mi2.shape)
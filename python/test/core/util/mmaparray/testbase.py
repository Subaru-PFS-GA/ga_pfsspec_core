from unittest import TestCase
import h5py

class testbase(TestCase):
    def get_test_hdf5_filename(self):
        return '/datascope/subaru/data/pfsspec/models/stellar/rbf/bosz/bosz_50000_FGK/pca-2/spectra.h5'

    def get_test_hdf5(self):
        fn = self.get_test_hdf5_filename()
        return h5py.File(fn, 'r')
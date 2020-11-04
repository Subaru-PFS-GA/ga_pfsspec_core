from test.test_base import TestBase
import h5py as h5
import os
import numpy as np

from pfsspec.data.dataset import Dataset

class TestDataset(TestBase):
    def get_test_dataset(self):
        ds = Dataset()
        ds.filename = os.path.join(self.PFSSPEC_TEST_PATH, 'test_dataset.h5')
        ds.fileformat = 'h5'

        return ds

    def load_large_dataset(self):
        filename = '/scratch/ceph/dobos/data/pfsspec/train/sdss_stellar_model/dataset/bosz/nowave_10k/train/dataset.h5'
        ds = Dataset(preload_arrays=False)
        ds.load(filename)
        return ds

    def test_init_storage(self):
        ds = self.get_test_dataset()
        ds.init_storage(4000, 10000, constant_wave=True)

        with h5.File(ds.filename, 'r') as f:
            self.assertEqual((4000,), f['wave'].shape)
            self.assertEqual((10000, 4000), f['flux'].shape)
            self.assertEqual((10000, 4000), f['error'].shape)
            self.assertEqual((10000, 4000), f['mask'].shape)

    def test_load(self):
        ds = self.load_large_dataset()

        self.assertEqual((10000, 25), ds.params.shape)
        self.assertTrue(ds.constant_wave)

        # fn = os.path.join(self.PFSSPEC_TEST_PATH, 'test_dataset.h5')
        # ds = Dataset(preload_arrays=True)
        # ds.load(fn)

        # self.assertEqual((4000,), ds.params.shape)

    def test_get_wave(self):
        ds = self.load_large_dataset()
        self.assertEqual((3823,), ds.get_wave(np.s_[:]).shape)

    def test_get_flux(self):
        ds = self.load_large_dataset()
        self.assertEqual((3823,), ds.get_flux(np.s_[0, :]).shape)

    def test_get_error(self):
        ds = self.load_large_dataset()
        self.assertEqual((3823,), ds.get_error(np.s_[0, :]).shape)

    def test_get_mask(self):
        ds = self.load_large_dataset()
        self.assertEqual((3823,), ds.get_mask(np.s_[0, :]).shape)

    def test_get_spectrum(self):
        ds = self.load_large_dataset()
        spec = ds.get_spectrum(123)

        self.assertEqual((3823,), spec.wave.shape)
        self.assertEqual((3823,), spec.flux.shape)

    def test_split(self):
        ds = self.load_large_dataset()
        split_index, a, b = ds.split(0.2)

    def test_where(self):
        ds = self.load_large_dataset()
        filter = (ds.params['snr'] > 30)
        a = ds.where(filter)
        b = ds.where(~filter)

        self.assertEqual(ds.params.shape[0], a.params.shape[0] + b.params.shape[0])

    def test_merge(self):
        ds = self.load_large_dataset()
        filter = (ds.params['snr'] > 30)
        a = ds.where(filter)
        b = ds.where(~filter)

        a = a.merge(b)

        self.assertEqual(ds.params.shape[0], a.params.shape[0])

    def test_filter(self):
        pass

        # ts_gen.set_filter(None)
        # print(ts_gen.get_input_count(), ts_gen.batch_size, ts_gen.steps_per_epoch())

        # ts_gen.set_filter(ts_gen.dataset.params['interp_param'] == 'Fe_H')
        # print(ts_gen.get_input_count(), ts_gen.batch_size, ts_gen.steps_per_epoch())

        # ts_gen.set_filter(None)
        # ts_gen.shuffle = False
        # ts_gen.reshuffle()
        # idx = ts_gen.get_batch_index(1)
        # print(idx)
        # print(ts_gen.get_batch(1)[1])

        # ts_gen.set_filter(ts_gen.dataset.params['interp_param'] == 'Fe_H')
        # ts_gen.shuffle = False
        # ts_gen.reshuffle()
        # idx = ts_gen.get_batch_index(1)
        # print(idx)
        # print(ts_gen.get_batch(1)[1])
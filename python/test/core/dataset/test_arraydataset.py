import h5py
import os
import numpy as np
import pandas as pd
from pfs.ga.pfsspec.core.dataset.spectrumdataset import SpectrumDataset

from test.core import TestBase
from pfs.ga.pfsspec.core import PfsObject
from pfs.ga.pfsspec.core import Spectrum
from pfs.ga.pfsspec.core.dataset import ArrayDataset
from pfs.ga.pfsspec.core.dataset import DatasetConfig

class TestArrayDataset(TestBase):
    #region Helper functions

    class SpectrumTestDataset(ArrayDataset):
        def __init__(self, config=None, preload_arrays=None, value_shape=None, orig=None):
            super(TestArrayDataset.SpectrumTestDataset, self).__init__(config, preload_arrays, orig=orig)
            self.value_shape = value_shape

        def init_values(self, row_count=None):
            super(TestArrayDataset.SpectrumTestDataset, self).init_values(row_count=row_count)

            self.init_value('flux', dtype=np.float)
            self.init_value('error', dtype=np.float)
            self.init_value('mask', dtype=np.int)

        def allocate_values(self, row_count):
            super(TestArrayDataset.SpectrumTestDataset, self).allocate_values(row_count=row_count)

            self.allocate_value('flux', self.value_shape, dtype=np.float)
            self.allocate_value('error', self.value_shape, dtype=np.float)
            self.allocate_value('mask', self.value_shape, dtype=np.int)

    def create_dataset(self, size=100, value_shape=5000):
        """
        Create a test dataset with a few random columns and spectra
        """
        
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=True, value_shape=value_shape)
        ds.params = pd.DataFrame({
            'id': np.arange(0, size, dtype=np.int),
            'Fe_H': np.random.normal(-1, 0.5, size=size),
            'T_eff': np.random.uniform(3500, 7000, size=size),
            'log_g': np.random.uniform(1.5, 4.5, size=size)
        })

        ds.allocate_values(row_count=size)

        ds.values['flux'][:] = 1e-17 * np.random.normal(1, 0.1, size=(size, value_shape))
        ds.values['error'][:] = 1e-17 * np.random.normal(1, 0.1, size=(size, value_shape))
        ds.values['mask'][:] = np.int32(np.random.normal(1.0, 1.0, size=(size, value_shape)) > 0)

        return ds

    def create_test_datasets(self, format='h5'):
        filename = self.get_test_filename(filename='small_dataset', ext=PfsObject.get_extension(format))
        if not os.path.isfile(filename):
            ds = self.create_dataset(size=100)
            ds.save(filename, format=format)

        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        if not os.path.isfile(filename):
            ds = self.create_dataset(size=10000)
            ds.save(filename, format=format)

    #endregion
    #region Counts and chunks

    def test_get_count(self):
        format = 'h5'
        self.create_test_datasets(format=format)

        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=False)
        ds.load(filename, format)
        self.assertEqual(10000, ds.get_count())

    def test_get_chunk_id(self):
        format = 'h5'
        self.create_test_datasets(format=format)

        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=False)

        self.assertEqual((0, 0), ds.get_chunk_id(0, 3000))
        self.assertEqual((1, 0), ds.get_chunk_id(3000, 3000))
        self.assertEqual((1, 300), ds.get_chunk_id(3300, 3000))

    def test_get_chunk_count(self):
        format = 'h5'
        self.create_test_datasets(format=format)

        filename = self.get_test_filename(filename='small_dataset', ext=PfsObject.get_extension(format))
        ds = ArrayDataset(preload_arrays=False)
        ds.load(filename, format)
        self.assertEqual(100, ds.get_count())
        self.assertEqual(10, ds.get_chunk_count(10))
        self.assertEqual(5, ds.get_chunk_count(23))
        self.assertEqual(1, ds.get_chunk_count(100))
        self.assertEqual(1, ds.get_chunk_count(1000))

    def test_get_chunk_slice(self):
        format = 'h5'
        self.create_test_datasets(format=format)

        filename = self.get_test_filename(filename='small_dataset', ext=PfsObject.get_extension(format))
        ds = ArrayDataset(preload_arrays=False)
        ds.load(filename, format)
        self.assertEqual(100, ds.get_count())

        self.assertEqual(slice(0, 10, None), ds.get_chunk_slice(10, 0))
        self.assertEqual(slice(90, 100, None), ds.get_chunk_slice(10, 9))

        self.assertEqual(slice(0, 23, None), ds.get_chunk_slice(23, 0))
        self.assertEqual(slice(69, 92, None), ds.get_chunk_slice(23, 3))
        self.assertEqual(slice(92, 100, None), ds.get_chunk_slice(23, 4))

        self.assertEqual(slice(0, 100, None), ds.get_chunk_slice(100, 0))
        
        self.assertEqual(slice(0, 100, None), ds.get_chunk_slice(1000, 0))

    #endregion
    #region Load and save

    def test_allocate_values_preload(self):
        size = 100

        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=True, value_shape=6000)
        ds.params = pd.DataFrame({
            'Fe_H': np.random.normal(-1, 0.5, size=size),
            'T_eff': np.random.uniform(3500, 7000, size=size),
            'log_g': np.random.uniform(1.5, 4.5, size=size)
        })
        ds.allocate_values(row_count=size)
        self.assertEqual((size, 6000), ds.values['flux'].shape)
        self.assertEqual((size, 6000), ds.values['error'].shape)
        self.assertEqual((size, 6000), ds.values['mask'].shape)

    def test_allocate_values_chunked(self):
        format = 'h5'
        size = 100
        filename = self.get_test_filename(ext=PfsObject.get_extension(format=format))

        if os.path.isfile(filename):
            os.remove(filename)
        
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=False, value_shape=6000)
        ds.params = pd.DataFrame({
            'Fe_H': np.random.normal(-1, 0.5, size=size),
            'T_eff': np.random.uniform(3500, 7000, size=size),
            'log_g': np.random.uniform(1.5, 4.5, size=size)
        })
        ds.save(filename, format=format)
        ds.allocate_values(size)
        with h5py.File(ds.filename, 'r') as f:
            self.assertEqual((size, 6000), f['flux'].shape)
            self.assertEqual((size, 6000), f['error'].shape)
            self.assertEqual((size, 6000), f['mask'].shape)

        if os.path.isfile(filename):
            os.remove(filename)

    def test_save_preload(self):
        for format in ['h5']: # , 'npz', 'numpy', 'pickle']:
            self.create_test_datasets(format=format)

    def test_load_preload(self):
        for format in ['h5']: # , 'npz', 'numpy', 'pickle']:
            self.create_test_datasets(format=format)

            filename = self.get_test_filename(filename='small_dataset', ext=PfsObject.get_extension(format))
            ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=True)
            ds.init_values()
            ds.load(filename, format)
            self.assertEqual((100, 4), ds.params.shape)
            self.assertEqual((100, 5000), ds.values['flux'].shape)
            self.assertEqual((100, 5000), ds.values['error'].shape)
            self.assertEqual((100, 5000), ds.values['mask'].shape)

    def test_load_chunked(self):
        format = 'h5'
        self.create_test_datasets(format=format)

        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=False)
        ds.init_values()
        ds.load(filename, format)
        self.assertEqual((10000, 4), ds.params.shape)
        self.assertIsNone(ds.values['flux'])
        self.assertEqual((5000,), ds.value_shapes['flux'])
        self.assertEqual(np.float, ds.value_dtypes['flux'])
        self.assertEqual(np.int, ds.value_dtypes['mask'])

    #endregion
    #region Params access

    def test_get_params_preload(self):
        format = 'h5'
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = ArrayDataset(preload_arrays=True)
        ds.load(filename, format)

        p = ds.get_params(names=None, idx=(), chunk_size=None, chunk_id=None)
        self.assertEqual((10000, 4), p.shape)

        p = ds.get_params(names=None, idx=slice(0, 10, None), chunk_size=None, chunk_id=None)
        self.assertEqual((10, 4), p.shape)

        p = ds.get_params(names=['Fe_H', 'T_eff'], idx=(), chunk_size=None, chunk_id=None)
        self.assertEqual((10000, 2), p.shape)

        p = ds.get_params(names=['Fe_H', 'T_eff'], idx=slice(0, 10, None), chunk_size=None, chunk_id=None)
        self.assertEqual((10, 2), p.shape)

    def test_get_params_chunked(self):
        format = 'h5'
        chunk_size = 300
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = ArrayDataset(preload_arrays=False)
        ds.load(filename, format)

        p = ds.get_params(names=None, idx=slice(0, 10, None), chunk_size=chunk_size, chunk_id=2)
        self.assertEqual((10, 4), p.shape)

        # Slice from the middle
        p = ds.get_params(names=['Fe_H', 'T_eff'], idx=(), chunk_size=chunk_size, chunk_id=2)
        self.assertEqual((chunk_size, 2), p.shape)

        # Last, partial slice
        p = ds.get_params(names=['Fe_H', 'T_eff'], idx=(), chunk_size=chunk_size, chunk_id=33)
        self.assertEqual((100, 2), p.shape)

        p = ds.get_params(names=['Fe_H', 'T_eff'], idx=slice(0, 10, None), chunk_size=chunk_size, chunk_id=2)
        self.assertEqual((10, 2), p.shape)

    def test_set_params(self):
        format = 'h5'
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = ArrayDataset(preload_arrays=True)
        ds.load(filename, format)

        v = np.random.normal(-1.0, 1.0, size=(10000))
        ds.set_params(names=['Fe_H'], values=v, idx=None, chunk_size=None, chunk_id=None)
        
        v = np.stack([
            np.random.normal(-1.0, 1.0, size=(100)),
            np.random.uniform(3500, 6500, size=(100))], axis=-1)
        idx = np.arange(10, 110, dtype=np.int)
        ds.set_params(names=['Fe_H', 'T_eff'], values=v, idx=idx, chunk_size=None, chunk_id=None)

        v = np.stack([
            np.random.normal(-1.0, 1.0, size=(100)),
            np.random.uniform(3500, 6500, size=(100))], axis=-1)
        idx = np.arange(10, 110, dtype=np.int)
        ds.set_params(names=['Fe_H', 'T_eff'], values=v, idx=idx, chunk_size=300, chunk_id=2)

    def test_append_params_row(self):
        ds = ArrayDataset()

        row = {
            'id': 0,
            'name': 'test1'
        }
        ds.append_params_row(row)

        row = {
            'id': 1,
            'name': 'test2'
        }
        ds.append_params_row(row)

    #endregion
    #region Array access with chunking and caching support

    def test_get_value_preload(self):
        format = 'h5'
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=True)
        ds.load(filename, format)

        self.assertIsNotNone(ds.values['flux'])
        self.assertIsNotNone(ds.values['error'])

        v = ds.get_value('flux', idx=slice(0, 10, None))
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual(np.float, v.dtype)

        v = ds.get_value('mask', idx=np.arange(20, 30, dtype=np.int))
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual(np.int, v.dtype)

    def test_get_value_nochunk(self):
        # No chunking + read directly from disk

        format = 'h5'
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=False)
        ds.load(filename, format)

        v = ds.get_value('flux', idx=slice(0, 10, None))
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual(np.float, v.dtype)

        v = ds.get_value('error', idx=np.arange(30, 20, -1, dtype=np.int))
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual(np.float, v.dtype)

        v = ds.get_value('mask', idx=np.arange(20, 30, dtype=np.int))
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual(np.int, v.dtype)

    def test_get_value_chunk(self):
        # Chunking + read cache

        format = 'h5'
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=False)
        ds.load(filename, format)

        # Cache is cold
        self.assertIsNone(ds.values['flux'])
        self.assertIsNone(ds.values['error'])

        # Read with cold cache
        v = ds.get_value('flux', idx=slice(0, 10, None), chunk_size=1000, chunk_id=0)
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual(np.float, v.dtype)
        self.assertIsNotNone(ds.values['flux'])
        self.assertIsNone(ds.values['error'])

        # Read with cold cache, reverse order indexing
        v = ds.get_value('error', idx=np.arange(30, 20, -1, dtype=np.int), chunk_size=1000, chunk_id=0)
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual(np.float, v.dtype)
        self.assertIsNotNone(ds.values['flux'])
        self.assertIsNotNone(ds.values['error'])
        
        # Read with hot cache
        v = ds.get_value('error', idx=slice(0, 10, None), chunk_size=1000, chunk_id=0)
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual(np.float, v.dtype)

        # Read partial chunk
        v = ds.get_value('error', idx=slice(0, 10, None), chunk_size=3000, chunk_id=3)
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual((1000, 5000), ds.values['error'].shape)
        self.assertEqual(np.float, v.dtype)

    def test_set_value_preload(self):
        # Write to memory

        format = 'h5'
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=True)
        ds.load(filename, format)

        self.assertIsNotNone(ds.values['flux'])
        self.assertIsNotNone(ds.values['error'])

        v = 1e-17 * np.random.normal(1.0, 0.1, size=(10, 5000))
        ds.set_value('flux', v, idx=slice(0, 10, None))

        v = 1e-17 * np.random.normal(1.0, 0.1, size=(10, 5000))
        ds.set_value('error', v, idx=np.arange(20, 30))

        # Reverse order indexing
        v = np.int32(np.random.normal(0.0, 0.1, size=(10, 5000)) > 0.2)
        ds.set_value('mask', v, idx=np.arange(30, 20, -1, dtype=np.int))

    def test_set_value_nochunk(self):
        # Write changes directly to disk

        format = 'h5'
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=False)
        ds.load(filename, format)

        self.assertIsNone(ds.values['flux'])
        self.assertIsNone(ds.values['error'])
        self.assertIsNone(ds.values['mask'])

        v = 1e-17 * np.random.normal(1.0, 0.1, size=(10, 5000))
        ds.set_value('flux', v, idx=slice(0, 10, None))

        v = 1e-17 * np.random.normal(1.0, 0.1, size=(10, 5000))
        ds.set_value('error', v, idx=np.arange(20, 30, dtype=np.int))

        # Reverse order indexing
        v = np.int32(np.random.normal(0.0, 0.1, size=(10, 5000)) > 0.2)
        ds.set_value('mask', v, idx=np.arange(30, 20, -1, dtype=np.int))

        self.assertIsNone(ds.values['flux'])
        self.assertIsNone(ds.values['error'])
        self.assertIsNone(ds.values['mask'])

    def test_set_value_chunk(self):
        # Use write back cache
        
        format = 'h5'
        self.create_test_datasets(format=format)
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=False)
        ds.load(filename, format)

        self.assertIsNone(ds.values['flux'])
        self.assertIsNone(ds.values['error'])
        self.assertIsNone(ds.values['mask'])
        self.assertFalse(ds.cache_dirty)

        # Write to the cache
        v = 1e-17 * np.random.normal(1.0, 0.1, size=(10, 5000))
        ds.set_value('flux', v, idx=slice(0, 10, None), chunk_size=1000, chunk_id=0)
        self.assertTrue(ds.cache_dirty)
        self.assertIsNotNone(ds.values['flux'])
        self.assertIsNone(ds.values['error'])

        # Write to the cache, params will be written to the disk directly
        v = 1e-17 * np.random.normal(1.0, 0.1, size=(10, 5000))
        ds.set_value('error', v, idx=slice(0, 10, None), chunk_size=1000, chunk_id=0)
        self.assertTrue(ds.cache_dirty)
        self.assertIsNotNone(ds.values['flux'])
        self.assertIsNotNone(ds.values['error'])

        # Force flush before writing another slice
        v = 1e-17 * np.random.normal(1.0, 0.1, size=(10, 5000))
        ds.set_value('error', v, idx=slice(0, 10, None), chunk_size=1000, chunk_id=1)
        self.assertTrue(ds.cache_dirty)
        self.assertIsNone(ds.values['flux'])
        self.assertIsNotNone(ds.values['error'])

    def test_reset_cache_all(self):
        # Chunking + read through cache

        format = 'h5'
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=False)
        ds.load(filename, format)

        # Read with cold cache
        v = ds.get_value('flux', idx=np.arange(30, 20, -1, dtype=np.int), chunk_size=1000, chunk_id=0)
        v = ds.get_value('error', idx=np.arange(30, 20, -1, dtype=np.int), chunk_size=1000, chunk_id=0)
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual(np.float, v.dtype)
        self.assertIsNotNone(ds.values['flux'])
        self.assertIsNotNone(ds.values['error'])

        # Read from different chunk, this forces a cache reset
        v = ds.get_value('error', idx=slice(0, 10, None), chunk_size=1000, chunk_id=1)
        self.assertEqual((10, 5000), v.shape)
        self.assertEqual(np.float, v.dtype)
        self.assertIsNone(ds.values['flux'])
        self.assertIsNotNone(ds.values['error'])

    #endregion

    def test_reset_index(self):
        format = 'h5'
        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        ds = TestArrayDataset.SpectrumTestDataset(preload_arrays=False)
        ds.load(filename, format)

        ds.reset_index(ds.params)

    # def test_split(self):
    #     ds = self.load_large_dataset()
    #     split_index, a, b = ds.split(0.2)

    # def test_where(self):
    #     ds = self.load_large_dataset()
    #     filter = (ds.params['snr'] > 30)
    #     a = ds.where(filter)
    #     b = ds.where(~filter)

    #     self.assertEqual(ds.params.shape[0], a.params.shape[0] + b.params.shape[0])

    # def test_merge(self):
    #     ds = self.load_large_dataset()
    #     filter = (ds.params['snr'] > 30)
    #     a = ds.where(filter)
    #     b = ds.where(~filter)

    #     a = a.merge(b)

    #     self.assertEqual(ds.params.shape[0], a.params.shape[0])

    # def test_filter(self):
    #     pass

    #     # ts_gen.set_filter(None)
    #     # print(ts_gen.get_input_count(), ts_gen.batch_size, ts_gen.steps_per_epoch())

    #     # ts_gen.set_filter(ts_gen.dataset.params['interp_param'] == 'Fe_H')
    #     # print(ts_gen.get_input_count(), ts_gen.batch_size, ts_gen.steps_per_epoch())

    #     # ts_gen.set_filter(None)
    #     # ts_gen.shuffle = False
    #     # ts_gen.reshuffle()
    #     # chunk_id, idx = ts_gen.get_batch_index(1)
    #     # print(idx)
    #     # print(ts_gen.get_batch(1)[1])

    #     # ts_gen.set_filter(ts_gen.dataset.params['interp_param'] == 'Fe_H')
    #     # ts_gen.shuffle = False
    #     # ts_gen.reshuffle()
    #     # chunk_id, idx = ts_gen.get_batch_index(1)
    #     # print(idx)
    #     # print(ts_gen.get_batch(1)[1])
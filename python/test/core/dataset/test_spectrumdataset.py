import h5py
import os
import numpy as np
import pandas as pd

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core import PfsObject
from pfs.ga.pfsspec.core.dataset import SpectrumDataset

class TestSpectrumDataset(TestBase):
    def create_dataset(self, spectrum_count=100, wave_count=6000, constant_wave=True):
        """
        Create a test dataset with a few random columns and spectra
        """
        
        ds = SpectrumDataset(constant_wave=constant_wave, preload_arrays=True)
        ds.allocate_values(spectrum_count, wave_count)

        ds.params = pd.DataFrame({
            'id': np.arange(0, spectrum_count, dtype=int),
            'Fe_H': np.random.normal(-1, 0.5, size=spectrum_count),
            'T_eff': np.random.uniform(3500, 7000, size=spectrum_count),
            'log_g': np.random.uniform(1.5, 4.5, size=spectrum_count)
        })

        if constant_wave:
            wave_edges = np.linspace(3000, 9000, wave_count + 1) 
            wave = 0.5 * (wave_edges[1:] + wave_edges[:-1])
            ds.set_wave(wave, wave_edges=wave_edges)
        else:
            wave_edges = np.stack(spectrum_count * [np.linspace(3000, 9000, wave_count + 1)])
            wave = 0.5 * (wave_edges[..., 1:] + wave_edges[..., :-1])
            ds.set_wave(wave, wave_edges=wave_edges)

        ds.set_flux(1e-17 * np.random.normal(1, 0.1, size=(spectrum_count, wave_count)))
        ds.set_error(1e-17 * np.random.normal(1, 0.1, size=(spectrum_count, wave_count)))
        ds.set_mask(np.int32(np.random.normal(1.0, 1.0, size=(spectrum_count, wave_count)) > 0))

        return ds

    def create_test_datasets(self, format='h5'):
        filename = self.get_test_filename(filename='small_dataset', ext=PfsObject.get_extension(format))
        if not os.path.isfile(filename):
            ds = self.create_dataset(spectrum_count=100)
            ds.save(filename, format=format)

        filename = self.get_test_filename(filename='small_wave_dataset', ext=PfsObject.get_extension(format))
        if not os.path.isfile(filename):
            ds = self.create_dataset(spectrum_count=100, constant_wave=False)
            ds.save(filename, format=format)

        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        if not os.path.isfile(filename):
            ds = self.create_dataset(spectrum_count=10000)
            ds.save(filename, format=format)

        filename = self.get_test_filename(filename='large_wave_dataset', ext=PfsObject.get_extension(format))
        if not os.path.isfile(filename):
            ds = self.create_dataset(spectrum_count=10000, constant_wave=False)
            ds.save(filename, format=format)

    def test_load_preload(self):
        for format in ['h5']: # , 'npz', 'numpy', 'pickle']:
            self.create_test_datasets(format=format)

            filename = self.get_test_filename(filename='small_dataset', ext=PfsObject.get_extension(format))
            ds = SpectrumDataset(preload_arrays=True)
            ds.load(filename, format)
            self.assertEqual((100, 4), ds.params.shape)
            self.assertTrue(ds.constant_wave)
            self.assertIsNotNone(ds.wave)
            self.assertEqual((100, 6000), ds.get_value_shape('flux'))
            self.assertEqual((100, 6000), ds.get_value_shape('error'))
            self.assertEqual((100, 6000), ds.get_value_shape('mask'))

            filename = self.get_test_filename(filename='small_wave_dataset', ext=PfsObject.get_extension(format))
            ds = SpectrumDataset(constant_wave=False, preload_arrays=True)
            ds.load(filename, format)
            self.assertEqual((100, 4), ds.params.shape)
            self.assertFalse(ds.constant_wave)
            self.assertIsNone(ds.wave)
            self.assertEqual((100, 6000,), ds.get_value_shape('wave'))
            self.assertEqual((100, 6000), ds.get_value_shape('flux'))
            self.assertEqual((100, 6000), ds.get_value_shape('error'))
            self.assertEqual((100, 6000), ds.get_value_shape('mask'))

    def test_get_spectrum(self):
        format = 'h5'
        filename = self.get_test_filename(filename='small_dataset', ext=PfsObject.get_extension(format))
        ds = SpectrumDataset(preload_arrays=True)
        ds.load(filename, format)
        
        spec = ds.get_spectrum(23)

        self.assertEqual((6000,), spec.wave.shape)
        self.assertEqual((6001,), spec.wave_edges.shape)
        self.assertEqual((6000,), spec.flux.shape)

    def test_get_spectrum_wave(self):
        format = 'h5'
        filename = self.get_test_filename(filename='small_wave_dataset', ext=PfsObject.get_extension(format))
        ds = SpectrumDataset(preload_arrays=True)
        ds.load(filename, format)
        
        spec = ds.get_spectrum(23)

        self.assertEqual((6000,), spec.wave.shape)
        self.assertEqual((2, 6000), spec.wave_edges.shape)
        self.assertEqual((6000,), spec.flux.shape)
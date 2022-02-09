class TestSpectrumDataset():
    def create_dataset(self, size=100, constant_wave=True):
        """
        Create a test dataset with a few random columns and spectra
        """
        
        ds = Dataset(preload_arrays = True)
        ds.params = pd.DataFrame({
            'Fe_H': np.random.normal(-1, 0.5, size=size),
            'T_eff': np.random.uniform(3500, 7000, size=size),
            'log_g': np.random.uniform(1.5, 4.5, size=size)
        })

        if constant_wave:
            ds.wave = np.linspace(3000, 9000, 6000)
        else:
            ds.wave = np.stack(size * [np.linspace(3000, 9000, 6000)])
        ds.flux = 1e-17 * np.random.normal(1, 0.1, size=(size, ds.wave.shape[-1]))
        ds.error = 1e-17 * np.random.normal(1, 0.1, size=(size, ds.wave.shape[-1]))
        ds.mask = np.int32(np.random.normal(1.0, 1.0, size=(size, ds.wave.shape[-1])) > 0)

        return ds

    def create_test_datasets(self, format='h5'):
        filename = self.get_test_filename(filename='small_dataset', ext=PfsObject.get_extension(format))
        if not os.path.isfile(filename):
            ds = self.create_dataset(size=100)
            ds.save(filename, format=format)

        filename = self.get_test_filename(filename='small_wave_dataset', ext=PfsObject.get_extension(format))
        if not os.path.isfile(filename):
            ds = self.create_dataset(size=100, constant_wave=False)
            ds.save(filename, format=format)

        filename = self.get_test_filename(filename='large_dataset', ext=PfsObject.get_extension(format))
        if not os.path.isfile(filename):
            ds = self.create_dataset(size=10000)
            ds.save(filename, format=format)

        filename = self.get_test_filename(filename='large_wave_dataset', ext=PfsObject.get_extension(format))
        if not os.path.isfile(filename):
            ds = self.create_dataset(size=10000, constant_wave=False)
            ds.save(filename, format=format)

    def test_load_preload(self):
        for format in ['h5']: # , 'npz', 'numpy', 'pickle']:
            self.create_test_datasets(format=format)

            filename = self.get_test_filename(filename='small_dataset', ext=PfsObject.get_extension(format))
            ds = Dataset(preload_arrays=True)
            ds.load(filename, format)
            self.assertEqual((100, 3), ds.params.shape)
            self.assertEqual((6000,), ds.wave.shape)
            self.assertTrue(ds.constant_wave)
            self.assertEqual((100, 6000), ds.flux.shape)
            self.assertEqual((100, 6000), ds.error.shape)
            self.assertEqual((100, 6000), ds.mask.shape)

            filename = self.get_test_filename(filename='small_wave_dataset', ext=PfsObject.get_extension(format))
            ds = Dataset(preload_arrays=True)
            ds.load(filename, format)
            self.assertEqual((100, 3), ds.params.shape)
            self.assertEqual((100, 6000,), ds.wave.shape)
            self.assertFalse(ds.constant_wave)
            self.assertEqual((100, 6000), ds.flux.shape)
            self.assertEqual((100, 6000), ds.error.shape)
            self.assertEqual((100, 6000), ds.mask.shape)

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
import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.obsmod.resampling import Binning

class TestBinning(TestBase):
    def test_find_wave_edges(self):
        wave_edges = np.linspace(3000, 9000, 6001)
        wave = 0.5 * (wave_edges[1:] + wave_edges[:-1])
        
        we = Binning.find_wave_edges(wave)
        self.assertEqual(we.shape, wave_edges.shape)

        we = Binning.find_wave_edges(wave, binning='lin')
        self.assertEqual(we.shape, wave_edges.shape)

        # TODO: test log binning

    def test_round_wave_min_max(self):
        wmin, wmax = 3000.1, 8999.9
        binsize = 0.5

        wmin, wmax = Binning.round_wave_min_max(wmin, wmax, binsize)
        self.assertEqual(wmin, 3000)
        self.assertEqual(wmax, 9000)

        wmin, wmax = Binning.round_wave_min_max(wmin, wmax, binsize, binning='lin')
        self.assertEqual(wmin, 3000)
        self.assertEqual(wmax, 9000)

        wmin, wmax = Binning.round_wave_min_max(wmin, wmax, binsize, binning='log')
        self.assertEqual(np.log(wmin), 8.0)
        self.assertEqual(np.log(wmax), 9.5)

    def test_generate_wave_bins(self):
        wmin, wmax = 3000, 9000

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, nbins=6000, binning='lin')
        self.assertEqual(wave.shape, (6000,))
        self.assertEqual(wave[0], 3000.5)
        self.assertEqual(wave[-1], 8999.5)
        self.assertEqual(wave_edges.shape, (6001,))
        self.assertEqual(wave_edges[0], 3000)
        self.assertEqual(wave_edges[-1], 9000)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, binsize=0.5, binning='lin')
        self.assertEqual(wave.shape, (12000,))
        self.assertEqual(wave[0], 3000.25)
        self.assertEqual(wave[-1], 8999.75)
        self.assertEqual(wave_edges.shape, (12001,))
        self.assertEqual(wave_edges[0], 3000)
        self.assertEqual(wave_edges[-1], 9000)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, nbins=6000, binning='log')
        self.assertEqual(wave.shape, (6000,))
        # self.assertEqual(wave[0], 3000.5)
        # self.assertEqual(wave[-1], 8999.5)
        self.assertEqual(wave_edges.shape, (6001,))
        self.assertAlmostEqual(wave_edges[0], 3000)
        self.assertAlmostEqual(wave_edges[-1], 9000)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, binsize=np.log(1.001), binning='log')
        self.assertEqual(wave.shape, (1100,))
        # self.assertEqual(wave[0], 3000.5)
        # self.assertEqual(wave[-1], 8999.5)
        self.assertEqual(wave_edges.shape, (1101,))
        self.assertAlmostEqual(wave_edges[0], 3000)
        self.assertAlmostEqual(wave_edges[-1], 9007.545861082044)
import numpy as np
import numpy.testing as npt

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

    def test_generate_wave_bins_align_edges(self):
        wmin, wmax = 3000, 9000

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, nbins=6000, binning='lin')
        self.assertTrue(np.all((wave < wave_edges[1:]) & (wave > wave_edges[:-1])))
        self.assertEqual(wave.shape, (6000,))
        npt.assert_almost_equal(wave[0], 3000.5)
        npt.assert_almost_equal(wave[-1], 8999.5)
        self.assertEqual(wave_edges.shape, (6001,))
        npt.assert_almost_equal(wave_edges[0], 3000.0)
        npt.assert_almost_equal(wave_edges[-1], 9000.0)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, binsize=0.5, binning='lin')
        self.assertTrue(np.all((wave < wave_edges[1:]) & (wave > wave_edges[:-1])))
        self.assertEqual(wave.shape, (12000,))
        npt.assert_almost_equal(wave[0], 3000.25)
        npt.assert_almost_equal(wave[-1], 8999.75)
        self.assertEqual(wave_edges.shape, (12001,))
        npt.assert_almost_equal(wave_edges[0], 3000.0)
        npt.assert_almost_equal(wave_edges[-1], 9000.0)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, nbins=6000, binning='log')
        self.assertTrue(np.all((wave < wave_edges[1:]) & (wave > wave_edges[:-1])))
        self.assertEqual(wave.shape, (6000,))
        npt.assert_almost_equal(wave[0], 3000.27466564)
        npt.assert_almost_equal(wave[-1], 8999.1760785)
        self.assertEqual(wave_edges.shape, (6001,))
        npt.assert_almost_equal(wave_edges[0], 3000.0)
        npt.assert_almost_equal(wave_edges[-1], 9000.0)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, binsize=np.log(1.001), binning='log')
        self.assertTrue(np.all((wave < wave_edges[1:]) & (wave > wave_edges[:-1])))
        self.assertEqual(wave.shape, (1100,))
        npt.assert_almost_equal(wave[0], 3001.49962519)
        npt.assert_almost_equal(wave[-1], 9003.045463168804)
        self.assertEqual(wave_edges.shape, (1101,))
        npt.assert_almost_equal(wave_edges[0], 3000.0)
        npt.assert_almost_equal(wave_edges[-1], 9007.54586108)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, resolution=30000, binning='log')
        self.assertTrue(np.all((wave < wave_edges[1:]) & (wave > wave_edges[:-1])))
        self.assertEqual(wave.shape, (32958,))
        npt.assert_almost_equal(wave[0], 3000.05000098)
        npt.assert_almost_equal(wave[-1], 8999.849999572161)
        self.assertEqual(wave_edges.shape, (32959,))
        npt.assert_almost_equal(wave_edges[0], 3000.0)
        npt.assert_almost_equal(wave_edges[-1], 9000.0)

    def test_generate_wave_bins_align_center(self):
        wmin, wmax = 3000, 9000

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, nbins=3001, binning='lin', align='center')
        self.assertTrue(np.all((wave < wave_edges[1:]) & (wave > wave_edges[:-1])))
        self.assertEqual(wave.shape, (3001,))
        npt.assert_almost_equal(wave[0], 3000.0)
        npt.assert_almost_equal(wave[-1], 9000.0)
        self.assertEqual(wave_edges.shape, (3002,))
        npt.assert_almost_equal(wave_edges[0], 2999.0)
        npt.assert_almost_equal(wave_edges[-1], 9001.0)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, binsize=2.0, binning='lin', align='center')
        self.assertTrue(np.all((wave < wave_edges[1:]) & (wave > wave_edges[:-1])))
        self.assertEqual(wave.shape, (3001,))
        npt.assert_almost_equal(wave[0], 3000.0)
        npt.assert_almost_equal(wave[-1], 9000.0)
        self.assertEqual(wave_edges.shape, (3002,))
        npt.assert_almost_equal(wave_edges[0], 2999.0)
        npt.assert_almost_equal(wave_edges[-1], 9001.0)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, nbins=6000, binning='log', align='center')
        self.assertTrue(np.all((wave < wave_edges[1:]) & (wave > wave_edges[:-1])))
        self.assertEqual(wave.shape, (6000,))
        npt.assert_almost_equal(wave[0], 3000.0)
        npt.assert_almost_equal(wave[-1], 9000.0)
        self.assertEqual(wave_edges.shape, (6001,))
        npt.assert_almost_equal(wave_edges[0], 2999.72531372)
        npt.assert_almost_equal(wave_edges[-1], 9000.8241343)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, binsize=np.log(1.001), binning='log', align='center')
        self.assertTrue(np.all((wave < wave_edges[1:]) & (wave > wave_edges[:-1])))
        self.assertEqual(wave.shape, (1101,))
        npt.assert_almost_equal(wave[0], 3000)
        npt.assert_almost_equal(wave[-1], 9007.54586108)
        self.assertEqual(wave_edges.shape, (1102,))
        npt.assert_almost_equal(wave_edges[0], 2998.501124063317)
        npt.assert_almost_equal(wave_edges[-1], 9012.048508631973)

        wave, wave_edges = Binning.generate_wave_bins(wmin, wmax, resolution=30000, binning='log', align='center')
        self.assertTrue(np.all((wave < wave_edges[1:]) & (wave > wave_edges[:-1])))
        self.assertEqual(wave.shape, (32960,))
        npt.assert_almost_equal(wave[0], 3000)
        npt.assert_almost_equal(wave[-1], 9000.0246105)
        self.assertEqual(wave_edges.shape, (32961,))
        npt.assert_almost_equal(wave_edges[0], 2999.95000125)
        npt.assert_almost_equal(wave_edges[-1], 9000.17460966)
from test.test_base import TestBase
import os

from pfsspec.data.dataset import Dataset

class TestDataset(TestBase):
    def test_split(self):
        ds = self.get_sdss_dataset()
        split_index, a, b = ds.split(0.2)

    def test_filter(self):
        ds = self.get_sdss_dataset()
        filter = (ds.params['snr'] > 60)
        a, b = ds.filter(filter)

        self.assertEqual(ds.flux.shape[0], a.flux.shape[0] + b.flux.shape[0])

    def test_merge(self):
        ds = self.get_sdss_dataset()
        filter = (ds.params['snr'] > 60)
        a, b = ds.filter(filter)

        a = a.merge(b)

        self.assertEqual(ds.flux.shape[0], a.flux.shape[0])

    def test_filter(self):
        pass

        # ts_gen.set_filter(None)
        # print(ts_gen.get_input_count(), ts_gen.batch_size, ts_gen.steps_per_epoch())

        # ts_gen.set_filter(ts_gen.dataset.params['interp_param'] == 'Fe_H')
        # print(ts_gen.get_input_count(), ts_gen.batch_size, ts_gen.steps_per_epoch())

        # ts_gen.set_filter(None)
        # ts_gen.shuffle = False
        # ts_gen.reshuffle()
        # idx = ts_gen.next_batch_index(1)
        # print(idx)
        # print(ts_gen.next_batch(1)[1])

        # ts_gen.set_filter(ts_gen.dataset.params['interp_param'] == 'Fe_H')
        # ts_gen.shuffle = False
        # ts_gen.reshuffle()
        # idx = ts_gen.next_batch_index(1)
        # print(idx)
        # print(ts_gen.next_batch(1)[1])
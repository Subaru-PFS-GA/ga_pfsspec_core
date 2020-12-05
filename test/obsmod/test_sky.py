from test.test_base import TestBase
import os

from pfsspec.obsmod.sky import Sky


class TestSky(TestBase):
    def test_load(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'noise/sky/r/sky.h5')
        sky = Sky()
        sky.preload_arrays = True
        sky.load(file, format='h5')

        self.assertIsNotNone(sky.wave)
        self.assertIsNotNone(sky.values['counts'])
        self.assertIsNotNone(sky.values['conv'])

        sky = Sky()
        sky.preload_arrays = False
        sky.load(file, format='h5')

        self.assertIsNotNone(sky.wave)
        data = sky.get_nearest_value('conv', za=10, fa=0.5)
        self.assertIsNotNone(data)

    def test_interpolate(self):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'noise/sky/r/sky.h5')
        sky = Sky()
        sky.preload_arrays = True
        sky.load(file, format='h5')

        data, _ = sky.interpolate_value_linear('conv', za=10, fa=0.5)
        self.assertIsNotNone(data)
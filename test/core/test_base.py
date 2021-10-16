import os
from unittest import TestCase
import matplotlib.pyplot as plt

class TestBase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.PFSSPEC_DATA_PATH = os.environ['PFSSPEC_DATA'].strip('"') if 'PFSSPEC_DATA' in os.environ else None
        cls.PFSSPEC_TEST_PATH = os.environ['PFSSPEC_TEST'].strip('"') if 'PFSSPEC_TEST' in os.environ else None

    def setUp(self):
        plt.figure(figsize=(10, 6))

    def get_filename(self, ext):
        filename = type(self).__name__[4:] + '_' + self._testMethodName[5:] + ext
        return filename

    def get_test_filename(self, filename=None, ext=None):
        if filename is None and ext is None:
            raise ValueError()
        elif filename is None:
            filename = self.get_filename(ext)
        elif ext is None:
            # Filename already contains the extensions
            pass
        else:
            filename += ext
        filename = os.path.join(self.PFSSPEC_TEST_PATH, filename)
        return filename

    def save_fig(self, f=None, filename=None):
        if f is None:
            f = plt
        if filename is None:
            filename = self.get_filename('.png')
        f.savefig(os.path.join(self.PFSSPEC_TEST_PATH, filename))
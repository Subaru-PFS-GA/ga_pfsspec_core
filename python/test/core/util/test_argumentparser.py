import os

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.util.argumentparser import ArgumentParser

class TestArgumentParser(TestBase):
    def test_get_config_files(self):
        parser = ArgumentParser()

        paths = parser._get_config_paths([])
        self.assertIsNone(paths)

        paths = parser._get_config_paths(['--other'])
        self.assertIsNone(paths)

        paths = parser._get_config_paths(['--other a b'])
        self.assertIsNone(paths)
        
        paths = parser._get_config_paths(['--config'])
        self.assertEqual([], paths)

        paths = parser._get_config_paths('--config a'.split(' '))
        self.assertEqual(['a'], paths)
        
        paths = parser._get_config_paths('--config a b'.split(' '))
        self.assertEqual(['a', 'b'], paths)

        paths = parser._get_config_paths('--config a b --other'.split(' '))
        self.assertEqual(['a', 'b'], paths)

        paths = parser._get_config_paths('--config a b -o'.split(' '))
        self.assertEqual(['a', 'b'], paths)

        paths = parser._get_config_paths('--config a b --other c d'.split(' '))
        self.assertEqual(['a', 'b'], paths)

        paths = parser._get_config_paths('--config --other c d'.split(' '))
        self.assertEqual([], paths)

        paths = parser._get_config_paths('--other c d --config'.split(' '))
        self.assertEqual([], paths)

        paths = parser._get_config_paths('--other c d --config a'.split(' '))
        self.assertEqual(['a'], paths)

        paths = parser._get_config_paths('--other c d --config a b'.split(' '))
        self.assertEqual(['a', 'b'], paths)

        paths = parser._get_config_paths('--other c d --config a b --other'.split(' '))
        self.assertEqual(['a', 'b'], paths)
import os

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core import PfsObject

class JsonTestHelperObject(PfsObject):
    def __init__(self):
        super(JsonTestHelperObject, self).__init__()
        self.data1 = 'test1'
        self.data2 = 12
        self.data3 = [0, 1, 2]
        self.data4 = {'A': 0, 'B': 1}

class TestPfsObject(TestBase):
    def test_save_json(self):
        filename = self.get_test_filename('.json')
        o = JsonTestHelperObject()
        o.save_json(filename)

        j = \
"""{
    "data1": "test1",
    "data2": 12,
    "data3": [
        0,
        1,
        2
    ],
    "data4": {
        "A": 0,
        "B": 1
    }
}"""

        with open(filename, 'r') as f:
            d = f.read()
            self.assertEqual(d, j)

    def test_load_json(self):
        j = \
"""{
    "data1": "test2",
    "data2": 24,
    "data3": [
        4,
        5,
        6
    ],
    "data4": {
        "C": 5,
        "D": 4
    }
}"""

        filename = self.get_test_filename('.json')
        with open(filename, 'w') as f:
            f.write(j)

        o = JsonTestHelperObject()
        o.load_json(filename)

        self.assertEqual(o.data1, 'test2')
        self.assertEqual(o.data2, 24)
        self.assertEqual(o.data3, [4, 5, 6])
        self.assertEqual(o.data4, {'C':5, 'D':4})
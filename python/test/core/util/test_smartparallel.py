import os

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.util.smartparallel import SmartParallel

class TestSmartParallel(TestBase):

    # Use a separate class for testing because anything inherited from
    # TestCase is not pickleable.
    class SmartParallelHelper():
        def initializer(self, worker_id):
            pass

        def worker(self, i):
            return i

        def map(self):
            res = []
            with SmartParallel(self.initializer, verbose=True, parallel=True) as pool:
                for i in pool.map(self.worker, range(100)):
                    res.append(i)
            return res
    
    def test_map_parallel(self):
        h = TestSmartParallel.SmartParallelHelper()
        res = h.map()
        self.assertEqual(100, len(res))
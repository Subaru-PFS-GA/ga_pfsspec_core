import os, sys
import logging

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.util.smartparallel import SmartParallel

class TestSmartParallel(TestBase):

    # Use a separate class for testing because anything inherited from
    # TestCase is not pickleable.
    class SmartParallelHelper():
        def initializer(self, worker_id):
            pass

        def worker(self, i):
            if i == 30:
                logging.info(f'Forcefully exiting worker with pid {os.getpid()} at iteration {i}.')
                sys.exit(-1)
            else:
                logging.info(f'Logging from worker with pid {os.getpid()} at iteration {i}.')
                return i
            
        def error(self, ex, i):
            return -1

        def map(self):
            res = []
            with SmartParallel(self.initializer, verbose=True, parallel=True, threads=4) as pool:
                for i in pool.map(self.worker, self.error, range(100)):
                    res.append(i)
            return res
    
    def test_map_parallel(self):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        
        logging.info('Test parallel logging.')
        
        h = TestSmartParallel.SmartParallelHelper()
        res = h.map()
        self.assertEqual(100, len(res))
        self.assertTrue(-1 in res)

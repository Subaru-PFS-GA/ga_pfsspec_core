import os, sys
import logging
import traceback
import multiprocessing
from multiprocessing import Manager
from .pool import Pool
import numpy as np
from tqdm import tqdm

class SmartParallelError(Exception):
    def __init__(self, type, exception, traceback):
        self.type = type
        self.exception = exception
        self.traceback = traceback

class IterableQueue():
    def __init__(self, queue, length):
        self.logger = logging.getLogger()
        self.queue = queue
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.length > 0:
            self.length -= 1
            o = self.queue.get()
            if isinstance(o, SmartParallelError):
                ex = o
                msg = "An error occured in a Smart Parallel worker process.\n\nOriginal {}".format(''.join(ex.traceback))
                raise ex.type(msg)
            else:
                return o
        else:
            raise StopIteration()
    
class SmartParallel():
    def __init__(self, initializer=None, verbose=False, parallel=True, threads=None):
        if threads is not None:
            self.cpus = threads
        elif 'SLURM_CPUS_PER_TASK' in os.environ:
            self.cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            self.cpus = multiprocessing.cpu_count() // 2
        
        self.logger = logging.getLogger()
        self.manager = None
        self.pool = None
        self.queue_out = None
        self.initializer = initializer
        self.verbose = verbose
        self.parallel = parallel

    def __enter__(self):
        if self.parallel:
            self.logger.debug("Starting parallel execution on {} CPUs.".format(self.cpus))

            # Match logging preferences of the parent process
            log_stdout = False
            log_stderr = False
            for h in self.logger.handlers:
                log_stdout |= isinstance(h, logging.StreamHandler) and h.stream.name == '<stdout>'
                log_stderr |= isinstance(h, logging.StreamHandler) and h.stream.name == '<stderr>'

            # TODO: generate random seed for each worker

            self.manager = Manager()            
            self.pool = Pool(processes=self.cpus,
                             preinitializer=SmartParallel.pool_preinitializer,
                             initializer=SmartParallel.pool_initializer,
                             initargs=(self.logger.level, log_stdout, log_stderr))
        else:
            self.logger.debug("Starting serial execution.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parallel:
            self.logger.debug("Joining worker processes.")
            self.pool.close()
            self.pool.join()
            self.manager.shutdown()
            self.logger.debug("Finished parallel execution.")
        else:
            self.logger.debug("Finished serial execution.")
        return False

    def __del__(self):
        pass

    @staticmethod
    def pool_preinitializer():
        random_seed = np.random.randint(0, np.iinfo(np.int_).max) % 0xFFFFFFFF
        return (random_seed,)

    @staticmethod
    def pool_initializer(random_seed, log_level, log_stdout, log_stderr):
        # logger = logging.getLogger()
        # logger.setLevel(log_level)
        # if log_stdout:
        #     handler = logging.StreamHandler(sys.stdout)
        #     handler.setLevel(log_level)
        #     #handler.setFormatter(formatter)
        #     logger.addHandler(handler)
        # if log_stderr:
        #     handler = logging.StreamHandler(sys.stderr)
        #     handler.setLevel(log_level)
        #     #handler.setFormatter(formatter)
        #     logger.addHandler(handler)

        if random_seed is not None:
            np.random.seed(random_seed)
            logging.debug("Re-seeded random state on pid {} with seed {}".format(os.getpid(), random_seed))

    @staticmethod
    def pool_worker(i, queue_out, worker, error, *args, obj=None, **kwargs):
        # This executes in the worker process
        try:
            if obj is not None:
                o = worker(obj, i, *args, **kwargs)
            else:
                o = worker(i, *args, **kwargs)
            queue_out.put(o)
        except Exception as e:
            queue_out.put(SmartParallelError(*sys.exc_info()[:2], traceback.format_exception(*sys.exc_info())))

    @staticmethod
    def pool_error_callback(ex):
        i, queue_out, worker, error = ex.args[0:4]
        args = ex.args[4:]
        kwargs = ex.kwds
        obj = kwargs.pop("obj", None)
        try:
            if obj is not None:
                o = error(obj, i, *args, **kwargs)
            else:
                o = error(ex, i, *args, **kwargs)
            queue_out.put(o)
        except Exception as e:
            queue_out.put(SmartParallelError(*sys.exc_info()[:2], traceback.format_exception(*sys.exc_info())))

    def map(self, worker, error, items, *args, obj=None, **kwargs):
        # This executes in the main process
        
        if self.parallel:
            self.queue_out = self.manager.Queue(1024)

            args = (self.queue_out, worker, error) + args
            kwargs = {**kwargs, 'obj': obj}

            for i in items:
                self.pool.apply_async(SmartParallel.pool_worker, args=(i,) + args, kwds=kwargs,
                                      error_callback=SmartParallel.pool_error_callback)

            m = IterableQueue(self.queue_out, len(items))
        else:
            if self.initializer is not None:
                self.initializer(0)
            if obj is not None:
                m = map(lambda i: worker(obj, i, *args, **kwargs), items)
            else:
                m = map(lambda i: worker(i, *args, **kwargs), items)

        if self.verbose:
            m = tqdm(m, total=len(items))

        return m
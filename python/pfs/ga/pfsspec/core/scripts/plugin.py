import os
import numpy as np

import pfs.ga.pfsspec.core.util as util
from pfs.ga.pfsspec.core import PfsObject

class Plugin(PfsObject):
    def __init__(self, orig=None, random_seed=None, random_state=None):
        super().__init__(orig=orig)

        if not isinstance(orig, Plugin):
            self.random_seed = random_seed
            self.random_state = random_state
            self.parallel = True
            self.threads = None
            self.top = None
            self.resume = False
        else:
            self.random_seed = random_seed or orig.random_seed
            self.random_state = random_state or orig.random_state
            self.parallel = orig.parallel
            self.threads = orig.threads
            self.top = orig.top
            self.resume = orig.resume

    def get_arg(self, name, old_value, args=None):
        args = args or self.args
        return util.args.get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        args = args or self.args
        return util.args.is_arg(name, args)
    
    def add_args(self, parser, config):
        parser.add_argument('--top', type=int, help='Stop after this many items.\n')
        parser.add_argument('--resume', action='store_true', help='Resume processing with existing output.\n')

    def init_from_args(self, config, args):
        # Only allow parallel if random seed is not set
        # It would be very painful to reproduce the same dataset with multiprocessing
        self.threads = self.get_arg('threads', self.threads, args)
        self.parallel = self.random_seed is None and (self.threads is None or self.threads > 1)
        if not self.parallel:
            self.logger.info('Dataset builder running in sequential mode.')
        self.top = self.get_arg('top', self.top, args)
        self.resume = self.get_arg('resume', self.resume, args)

    def init_random_state(self):
        if self.random_state is None:
            if self.random_seed is not None:
                # NOTE: this seed won't be consistent across runs because pids can vary
                self.random_state = np.random.RandomState(self.random_seed + os.getpid() + 1)
            else:
                self.random_state = np.random.RandomState(None)
            
            self.logger.debug("Initialized random state on pid {}, rnd={}".format(os.getpid(), self.random_state.rand()))
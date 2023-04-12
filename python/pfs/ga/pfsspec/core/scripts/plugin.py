import os
import numpy as np

import pfs.ga.pfsspec.core.util as util
from pfs.ga.pfsspec.core import PfsObject

class Plugin(PfsObject):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, Plugin):
            self.config = None
            self.args = None
            self.parallel = True
            self.threads = None
            self.debug = False
            self.trace = None
            self.top = None
            self.resume = False
            self.verbose = False
        else:
            self.config = orig.config
            self.args = orig.args
            self.parallel = orig.parallel
            self.threads = orig.threads
            self.debug = orig.debug
            self.trace = orig.trace
            self.top = orig.top
            self.resume = orig.resume
            self.verbose = orig.verbose

    def get_arg(self, name, old_value, args=None):
        args = args or self.args
        return util.args.get_arg(name, old_value, args)

    def is_arg(self, name, args=None):
        args = args or self.args
        return util.args.is_arg(name, args)
    
    def add_args(self, parser, config):
        parser.add_argument('--top', type=int, help='Stop after this many items.\n')
        parser.add_argument('--resume', action='store_true', help='Resume processing with existing output.\n')

    def init_from_args(self, script, config, args):
        self.config = config
        self.args = args

        # Only allow parallel if random seed is not set
        # It would be very painful to reproduce the same dataset with multiprocessing
        self.threads = self.get_arg('threads', self.threads, args)
        self.parallel = self.random_seed is None and (self.threads is None or self.threads > 1)
        if not self.parallel:
            self.logger.info(f'Script plugin `{type(self).__name__}` running in sequential mode.')
        
        self.top = self.get_arg('top', self.top, args)
        self.resume = self.get_arg('resume', self.resume, args)

        if script is not None:
            self.debug = script.debug
            if script.trace:
                self.trace = self.create_trace(outdir=script.outdir, plot_level=script.trace_plot_level)
        else:
            self.debug = self.get_arg('debug', self.debug, args)

    def create_trace(self, outdir=None, plot=None):
        return None
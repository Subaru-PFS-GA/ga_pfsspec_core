import os
import numpy as np

from pfs.ga.pfsspec.core.plotting import SpectrumPlot, DistributionPlot, styles
from pfs.ga.pfsspec.core import Trace
from pfs.ga.pfsspec.core.util.args import *

class SpectrumStackerTrace(Trace):

    def __init__(self,
                 id=None,
                 figdir='.', logdir='.',
                 plot_inline=False, 
                 plot_level=Trace.PLOT_LEVEL_NONE, 
                 log_level=Trace.LOG_LEVEL_NONE):
        
        super().__init__(id=id,
                         figdir=figdir, logdir=logdir,
                         plot_inline=plot_inline, 
                         plot_level=plot_level,
                         log_level=log_level)

        ##

        self.reset()

    def reset(self):
        pass

    def add_args(self, config, parser):
        super().add_args(config, parser)
    
    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

        ##

    
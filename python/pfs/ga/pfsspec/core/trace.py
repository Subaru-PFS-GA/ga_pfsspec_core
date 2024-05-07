import os
import numpy as np

from pfs.ga.pfsspec.core.util.args import *
from pfs.ga.pfsspec.core.plotting import DiagramPage

class Trace():
    PLOT_LEVEL_NONE = 0
    PLOT_LEVEL_INFO = 1
    PLOT_LEVEL_DEBUG = 2
    PLOT_LEVEL_TRACE = 3

    LOG_LEVEL_NONE = 0
    LOG_LEVEL_INFO = 1
    LOG_LEVEL_DEBUG = 2

    def __init__(self, figdir='.', logdir='.',
                 plot_inline=False,
                 plot_level=PLOT_LEVEL_NONE,
                 log_level=LOG_LEVEL_NONE):
        
        # TODO: add inline (notebook) and file option

        self.figdir = figdir
        self.logdir = logdir
        self.create_outdir = True
        self.plot_inline = plot_inline
        self.plot_level = plot_level
        self.log_level = log_level

        self.diagram_pages = {}
        self.figure_formats = [ '.png' ]

    def add_args(self, config, parser):
        pass

    def init_from_args(self, script, config, args):
        self.plot_level = get_arg('plot_level', self.plot_level, args)

    def make_outdir(self, fn):
        dir = os.path.dirname(fn)
        if self.create_outdir and not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    def get_diagram_page(self, key, npages=1,  nrows=1, ncols=1, page_size=None, diagram_size=None):
        if key is not None and key in self.diagram_pages:
            return self.diagram_pages[key]
        else:
            f = DiagramPage(npages, nrows, ncols, page_size=page_size, diagram_size=diagram_size)
            self.diagram_pages[key] = f
            return f
        
    def format_figures(self):
        for k, fig in self.diagram_pages.items():
            fig.format()

    def show_figures(self):
        for k, fig in self.diagram_pages.items():
            fig.show()

    def save_figures(self):
        for k, fig in self.diagram_pages.items():
            for format in self.figure_formats:
                fn = os.path.join(self.figdir, k + format)
                self.make_outdir(fn)
                fig.save(fn)

    def flush_figures(self):
        self.format_figures()
        if self.plot_inline:
            self.show_figures()
        else:
            self.save_figures()
        self.diagram_pages = {}
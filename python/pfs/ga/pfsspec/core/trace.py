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

    def __init__(self, id=None,
                 figdir='.', logdir='.',
                 plot_inline=False,
                 plot_level=PLOT_LEVEL_NONE,
                 log_level=LOG_LEVEL_NONE):
        
        # TODO: add inline (notebook) and file option

        self.figdir = figdir                    # Directory where figures are saved
        self.logdir = logdir                    # Directory where logs are saved
        self.create_outdir = True               # Create output directories if they do not exist
        self.plot_inline = plot_inline          # Show plots inline in the notebook
        self.plot_level = plot_level            # Level of plot output
        self.log_level = log_level              # Level of log output

        self.diagram_pages = {}                 
        self.figure_formats = [ '.png' ]

        self.id = id if id is not None else ''  # Unique identifier of the object being processed

        self.reset()

    def reset(self):
        self.counters = {}

    def update(self, figdir=None, logdir=None):
        self.figdir = figdir if figdir is not None else self.figdir
        self.logdir = logdir if logdir is not None else self.logdir

    def add_args(self, config, parser):
        pass

    def init_from_args(self, script, config, args):
        self.plot_level = get_arg('plot_level', self.plot_level, args)

    def inc_counter(self, key):
        if key not in self.counters:
            self.counters[key] = 0
        self.counters[key] += 1

    def get_counter(self, key):
        if key not in self.counters:
            return None
        else:
            return self.counters[key]

    def make_outdir(self, fn):
        dir = os.path.dirname(fn)
        if self.create_outdir and not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    def get_diagram_page(self, key, npages=1, nrows=1, ncols=1, 
                         title=None,
                         page_size=None, diagram_size=None):
        
        # Substitute tokens
        key = key.format(id=self.id)
        title = title.format(id=self.id) if title is not None else None
        
        if key is not None and key in self.diagram_pages:
            return self.diagram_pages[key]
        else:
            f = DiagramPage(npages, nrows, ncols,
                            title=title,
                            page_size=page_size, diagram_size=diagram_size)
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

    def close_figures(self):
        for k, fig in self.diagram_pages.items():
            del fig
        self.diagram_pages = {}

    def flush_figures(self):
        self.format_figures()
        if self.plot_inline:
            self.show_figures()
        else:
            self.save_figures()
        self.close_figures()
import os
import numpy as np
import matplotlib.pyplot as plt

class Trace():
    PLOT_LEVEL_NONE = 0
    PLOT_LEVEL_INFO = 1
    PLOT_LEVEL_DEBUG = 2
    PLOT_LEVEL_TRACE = 3

    LOG_LEVEL_NONE = 0
    LOG_LEVEL_INFO = 1
    LOG_LEVEL_DEBUG = 2

    def __init__(self, outdir='.', plot_inline=False, plot_level=PLOT_LEVEL_NONE, log_level=LOG_LEVEL_NONE):
        # TODO: add inline (notebook) and file option

        self.outdir = outdir
        self.create_outdir = True
        self.plot_inline = plot_inline
        self.plot_level = plot_level
        self.log_level = log_level

        self.figure_size = (3.4, 2.5)
        self.figure_subplotsize = (3.4, 2.0)
        self.figure_dpi = 240
        self.figures = {}

    def make_outdir(self, fn):
        dir = os.path.dirname(fn)
        if self.create_outdir and not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    def get_figure(self, key, nrows=1, ncols=1):
        if key is not None and key in self.figures:
            return self.figures[key]
        else:
            figsize = (
                self.figure_size[0] + (ncols - 1) * self.figure_subplotsize[0],
                self.figure_size[1] + (nrows - 1) * self.figure_subplotsize[1]
            )
            f = plt.figure(figsize=figsize, dpi=self.figure_dpi)
            gs = f.add_gridspec(nrows=nrows, ncols=ncols)
            ax = np.empty((ncols, nrows), dtype=object)
            for r in range(nrows):
                for c in range(ncols):
                    ax[c, r] = f.add_subplot(gs[r, c])
            ax = ax.squeeze()
            if ax.size == 1:
                ax = ax.item()
            self.figures[key] = (f, ax)
            return f, ax
        
    def format_figures(self):
        for k, (f, axs) in self.figures.items():
            for ax in np.atleast_1d(axs).ravel():
                ax.legend()
            f.tight_layout()

    def show_figures(self):
        for k, (f, ax) in self.figures.items():
            f.show()

    def save_figures(self):
        for k, (f, ax) in self.figures.items():
            fn = os.path.join(self.outdir, k + '.png')
            self.make_outdir(fn)
            f.savefig(fn)

    def flush_figures(self):
        self.format_figures()
        if self.plot_inline:
            self.show_figures()
        else:
            self.save_figures()
        self.figures = {}
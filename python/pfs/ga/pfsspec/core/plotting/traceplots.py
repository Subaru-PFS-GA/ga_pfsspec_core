import os
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure

class TracePlots():
    PLOT_LEVEL_NONE = 0
    PLOT_LEVEL_INFO = 1
    PLOT_LEVEL_DEBUG = 2

    def __init__(self, outdir='.', plot_level=PLOT_LEVEL_NONE):
        # TODO: add inline (notebook) and file option

        self.outdir = outdir
        self.plot_level = plot_level

        self.figure_size = (3.4, 2.5)
        self.figure_subplotsize = (3.4, 2.0)
        self.figure_dpi = 240
        self.figures = {}

    def get_figure(self, key, nrows=1, ncols=1):
        if key is not None and key in self.figures:
            return self.figures[key]
        else:
            figsize = (
                self.figure_size[0] + (ncols - 1) * self.figure_subplotsize[0],
                self.figure_size[1] + (nrows - 1) * self.figure_subplotsize[1]
            )
            f = Figure(figsize=figsize, dpi=self.figure_dpi)
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

    def save_figures(self):
        for k, (f, ax) in self.figures.items():
            fn = os.path.join(self.outdir, k + '.png')
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            f.savefig(fn)

    def flush_figures(self):
        self.figures = {}
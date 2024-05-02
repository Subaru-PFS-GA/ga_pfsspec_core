import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Callable

from pfs.ga.pfsspec.core.sampling import Parameter, Distribution
from .diagram import Diagram
from .diagramaxis import DiagramAxis
from . import styles

class PriorPlot(Diagram):
    def __init__(self, ax: plt.Axes = None, diagram_axes=None,
                 title=None,
                 orig=None):

        if diagram_axes is None:
            diagram_axes = [ DiagramAxis(label='x'), DiagramAxis(label='p(x)') ]

        super().__init__(ax, title=title, diagram_axes=diagram_axes, orig=orig)

        self._validate()

    def _validate(self):
        pass

    def plot_prior(self, param_0, param_bounds, param_prior, param_step, normalize=False, **kwargs):
        # Figure out the plotting range
        if param_bounds is not None:
            pmin = param_bounds[0]
            pmax = param_bounds[1]
        elif param_step is not None:
            if param_0 is not None:
                pmin = param_0 - 5 * param_step
                pmax = param_0 + 5 * param_step
            else:
                pmin = -5 * param_step
                pmax = 5 * param_step
        else:
            pmin = -5
            pmax = 5

        # Make sure param_0 is within the plotting range
        if param_0 is not None:
            pmin = min(pmin, param_0)
            pmax = max(pmax, param_0)

        # Evaluate and plot the distribution
        if param_prior is not None:
            x = np.linspace(pmin, pmax, 300)

            if isinstance(param_prior, Distribution):
                y = param_prior.pdf(x)
            elif isinstance(param_prior, Callable):
                y = param_prior(x)
            else:
                raise NotImplementedError()
            
            if normalize:
                y /= np.nanmax(y)
            
            s = styles.thin_line(**kwargs)
            self._ax.plot(x, y, **s)

        # Buffer the range
        pmin -= (pmax - pmin) * 0.05
        pmax += (pmax - pmin) * 0.05

        # Plot param_0
        if param_0 is not None:
            s = styles.red_line(**styles.thin_line(**kwargs))
            self._ax.axvline(param_0, **s)

        self.update_limits(0, (pmin, pmax))

        self.apply()
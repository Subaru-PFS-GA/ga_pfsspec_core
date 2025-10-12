from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ..util import ReadOnlyList
from .diagram import styles
from .diagram import Diagram
from .diagramaxis import DiagramAxis
from .distributionplot import DistributionPlot

class CornerPlot():
    def __init__(self, 
                 diagram_page,
                 diagram_axes,
                 title=None):
        
        self.__diagram_page = diagram_page
        self.__diagram_axes = diagram_axes
        self.__title = title

        self.__diagrams = None
        self.__ax = None

        self.__create_diagrams()

    def __get_axes(self):
        return ReadOnlyList(self.__diagram_axes)

    axes = property(__get_axes)

    def __get_title(self):
        return self.__title
    
    def __set_title(self, value):
        self.__title = value

    title = property(__get_title, __set_title)

    def __create_diagrams(self):
        self.__diagrams = {}
        self.__ax = {}

        for i, iaxis in enumerate(self.__diagram_axes):
            for j, jaxis in enumerate(self.__diagram_axes):
                if i == j:
                    # Histogram, prior along the diagonal
                    d = DistributionPlot(diagram_axes=[iaxis, DiagramAxis((0, None), label='p(x)')])
                elif i < j:
                    d = Diagram(diagram_axes=[iaxis, jaxis])
                else:
                    continue

                self.__diagrams[(i, j)] = d

                ax = self.__diagram_page.add_diagram((j, i), d)
                self.__ax[(i, j)] = ax

    def update_limits(self, i, limits):
        """
        When generating multiple subplots, keep track of data min and max (or quantiles).
        This is used as a smarter version of sharex and sharey
        """

        for j, stat in enumerate((min, max)):
            if self.__diagram_axes[i].limits[j] is None:
                self.__diagram_axes[i].limits[j] = limits[j]
            else:
                self.__diagram_axes[i].limits[j] = stat(limits[j], self.__diagram_axes[i].limits[j])

    def apply(self):
        for d in self.__diagrams.values():
            if d is not None:
                d.apply()

        for i, iaxis in enumerate(self.__diagram_axes):
            for j, jaxis in enumerate(self.__diagram_axes):
                if (i, j) in self.__ax:
                    ax = self.__ax[(i, j)]

                    if i == j:
                        # Move the y-labels to the right side
                        ax.yaxis.set_label_position("right")
                        ax.yaxis.tick_right()

                        # Turn off the x-labels for all but the bottom plot
                        if i < len(self.__diagram_axes) - 1:
                            ax.set_xticklabels([])
                            ax.set_xlabel(None)
                    else:
                        # Turn off labels for inner plots
                        if i > 0:
                            ax.set_yticklabels([])
                            ax.set_ylabel(None)

                        if j < len(self.__diagram_axes) - 1:
                            ax.set_xticklabels([])
                            ax.set_xlabel(None)

                    # Turn on ticks on on all four axes and move them the inside
                    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, left=True, bottom=True)

        self.__diagram_page.apply()

    def scatter(self, *args, **kwargs):
        pass

    def plot(self):
        pass

    def errorbar(self, *args, sigma=None, **kwargs):
        # args: list of (x), (x, xerr) or (x, xerr_low, x_err_high) tuples for each
        #       axis

        if sigma is None:
            sigma = [ 1 ]

        def get_values(x):
            if not isinstance(x, (list, tuple)):
                x = (x,)

            if len(x) == 1:
                v = np.atleast_1d(x[0])
                err = None
            elif len(x) == 2:
                v = np.atleast_1d(x[0])
                err = np.atleast_1d(x[1])
            elif len(x) == 3:
                v = np.atleast_1d(x[0])
                err = np.stack(np.atleast_1d(x[1]), np.atleast_1d(x[2]))
            else:
                raise ValueError('Invalid number of arguments')
            
            return v, err
        
        for i, iaxis in enumerate(self.__diagram_axes):
            for j, jaxis in enumerate(self.__diagram_axes):
                if i == j:
                    ax = self.__ax[(i, i)]

                    x, xerr = get_values(args[i])

                    if x is not None:
                        s = styles.red_line(**styles.thin_line(**kwargs))
                        ax.axvline(x, **s)

                        if xerr is not None:
                            s = styles.dashed_line(**s)
                            for sig in sigma:
                                ax.axvline(x - sig * xerr, **s)
                                ax.axvline(x + sig * xerr, **s)
                elif i < j:
                    d = self.__diagrams[(i, j)]
                    ax = self.__ax[(i, j)]

                    x, xerr = get_values(args[i])
                    y, yerr = get_values(args[j])

                    d.errorbar(ax, x, y, xerr=xerr, yerr=yerr, **kwargs)

        self.apply()

    def __proj_cov(self, mu, cov, indexes):
        # Project the covariance matrix to the coordinates of  the plots

        # Permuted projection matrix
        proj = np.zeros_like(cov)
        for i1, i2 in enumerate(indexes):
            proj[i2, i1] = 1.0

        return mu[indexes], (proj.T @ cov @ proj)[:len(indexes),:len(indexes)]

    def __plot_cov_ellipse(self, ax, mu, cov, **kwargs):
        lam, v = np.linalg.eigh(cov)
        alpha = np.degrees(np.arctan2(*v[:, 0][::-1]))
        ellipse = Ellipse(mu, width=np.sqrt(lam[0]) * 2, height=np.sqrt(lam[1]) * 2, angle=alpha,
                          fill=False, **kwargs)
        ellipse.set_transform(ax.transData)
        ax.add_patch(ellipse)
        return ellipse

    def plot_covariance(self, mu, cov, sigma=None, **kwargs):
        # Plot the projections of a covariance matrix

        if sigma is None:
            sigma = [ 1 ]
        elif not isinstance(sigma, Iterable):
            sigma = [ sigma ]

        for i, iaxis in enumerate(self.__diagram_axes):
            for j, jaxis in enumerate(self.__diagram_axes):
                if i < j:
                    d = self.__diagrams[(i, j)]
                    ax = self.__ax[(i, j)]
                    
                    pmu, pcov = self.__proj_cov(mu, cov, [i, j])

                    for s in sigma:
                        self.__plot_cov_ellipse(ax, pmu, s**2 * pcov, **kwargs)

        self.apply()

    def plot_priors(self, *args, normalize=True, **kwargs):
        # Plot the priors along the diagonal

        for i, iaxis in enumerate(self.__diagram_axes):
            d = self.__diagrams[(i, i)]
            ax = self.__ax[(i, i)]

            param_prior, param_bounds, param_0, param_step = args[i]
            d.plot_prior(param_prior, param_bounds, param_0, param_step, 
                         normalize=normalize, auto_limits=False,
                         **kwargs)

        self.apply()

    def print_parameters(self, *args, **kwargs):
        # Print the parameters along the diagonal
        text = ''

        # TODO: add axis unit and number format

        for i, iaxis in enumerate(self.__diagram_axes):
            ax = self.__diagram_axes[i]
            param_fit, param_err = args[i]
            text += f'{ax.label} = ${param_fit:0.2f} \\pm {param_err:0.2f}$\n'

        ax = self.__diagram_page.page_ax[0]
        ax.text(1.0, 1.0, text, ha='right', va='top', transform=ax.transAxes)

    def hist(self):
        pass
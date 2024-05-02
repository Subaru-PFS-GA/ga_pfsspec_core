from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ..util import ReadOnlyList
from .diagram import Diagram

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
                    # TODO: histogram, prior along the diagonal
                    # d = Diagram(diagram_axes=[iaxis, jaxis])
                    # self.__diagram_page.add_diagram((i, j), d)
                    d = None
                    ax = None
                    pass
                elif i < j:
                    d = Diagram(diagram_axes=[iaxis, jaxis])
                    ax = self.__diagram_page.add_diagram((j, i), d)
                else:
                    continue

                self.__diagrams[(i, j)] = d
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

    def scatter(self, *args, **kwargs):
        pass

    def plot(self):
        pass

    def errorbar(self, *args, **kwargs):
        # args: list of (x), (x, xerr) or (x, xerr_low, x_err_high) tuples for each
        #       axis

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
                if i < j:
                    d = self.__diagrams[(i, j)]
                    ax = self.__ax[(i, j)]

                    x, xerr = get_values(args[i])
                    y, yerr = get_values(args[j])

                    d.errorbar(ax, x, y, xerr=xerr, yerr=yerr, **kwargs)

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

    def covariance(self, mu, cov, sigma=None, **kwargs):
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

    def hist(self):
        pass
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..util import *
from ..util.copy import *
from . import styles
from .diagramaxis import DiagramAxis

class Diagram():
    def __init__(self, ax: plt.Axes = None,
                 title=None,
                 diagram_axes=None, orig=None):

        if not isinstance(orig, Diagram):
            if diagram_axes is None:
                diagram_axes = [DiagramAxis(), DiagramAxis()]
            
            self.__diagram_axes = diagram_axes

            self.__title = title
        else:
            self.__diagram_axes = diagram_axes or safe_deep_copy(orig.__diagram_axes)

            self.__title = title or orig.__title

        # matplotlib axes
        self._ax = ax
        self._ax2 = None

    def __get_axes(self):
        return ReadOnlyList(self.__diagram_axes)

    axes = property(__get_axes)

    def __get_title(self):
        return self.__title
    
    def __set_title(self, value):
        self.__title = value

    title = property(__get_title, __set_title)

    def update_limits(self, i, limits):
        """
        When generating multiple subplots, keep track of data min and max (or quantiles).
        This is used as a smarter version of sharex and sharey
        """

        for j, stat in enumerate((min, max)):
            if self.__diagram_axes[i].limits[j] is None:
                self.__diagram_axes[i].limits[j] = limits[j]
            elif limits[j] is not None:
                self.__diagram_axes[i].limits[j] = stat(limits[j], self.__diagram_axes[i].limits[j])

    def apply(self, xlim=None, ylim=None, **kwargs):
        xlim = xlim if xlim is not None else self.__diagram_axes[0].limits
        ylim = ylim if ylim is not None else self.__diagram_axes[1].limits

        self._ax.tick_params(axis='both', which='major', **styles.tick_label_font())

        if xlim[0] != None or xlim[1] != None:
            self._ax.set_xlim(xlim)
        self._ax.set_xlabel(self.__diagram_axes[0].label, **styles.axis_label_font())
        if self.__diagram_axes[0].invert and not self._ax.xaxis_inverted():
            self._ax.invert_xaxis()
        self._ax.xaxis.offsetText.set(**styles.axis_label_font())

        if ylim[0] is not None or ylim[1] is not None:
            self._ax.set_ylim(ylim)
        self._ax.set_ylabel(self.__diagram_axes[1].label, **styles.axis_label_font())
        if self.__diagram_axes[1].invert and not self._ax.yaxis_inverted():
            self._ax.invert_yaxis()
        self._ax.yaxis.offsetText.set(**styles.axis_label_font())

        self._ax.set_title(self.__title, **styles.plot_title_font())
        
    def scatter(self, ax: plt.Axes, x, y, mask=None, s=None, **kwargs):
        style = styles.tiny_dots_scatter(**kwargs)
        
        s = s if s is not None else np.s_[:]
        mask = mask if mask is not None else np.s_[:]
        
        color = style.pop('color', style.pop('c', None))
        if isinstance(color, np.ndarray):
            color = color[mask][s]

        size = style.pop('size', style.pop('s', None))
        if isinstance(size, np.ndarray):
            size = size[mask][s]

        style = styles.sanitize_style(**style)

        l = ax.scatter(x[mask][s], y[mask][s], c=color, s=size, **style)
        self.apply(ax)

        return l

    def plot(self, ax: plt.Axes, x, y, fmt=None, mask=None, s=None, zorder=None, **kwargs):
        style = styles.tiny_dots_plot(**kwargs)

        s = s if s is not None else np.s_[:]
        mask = mask if mask is not None else np.s_[:]

        args = (x[mask][s], y[mask][s])
        if fmt is not None:
            args += (fmt,)
        
        l = ax.plot(*args, zorder=zorder, **styles.sanitize_style(**style))
        self.apply()

        return l
    
    def errorbar(self, ax: plt.Axes, x, y, xerr=None, yerr=None, fmt=None, mask=None, s=None, **kwargs):
        style = styles.thin_line(**kwargs)

        s = s if s is not None else np.s_[:]
        mask = mask if mask is not None else np.s_[:]

        def mask_vector(vector):
            if vector is None:
                return None
        
            if hasattr(vector, '__getitem__'):
                vector = vector[mask]
            
            if hasattr(vector, '__getitem__'):
                vector = vector[s]
            
            return vector

        # NOTE: xerr and yerr are swapped wrt. x and y!
        args = (mask_vector(x), mask_vector(y), mask_vector(yerr), mask_vector(xerr))
        
        if fmt is not None:
            args += (fmt,)

        l = ax.errorbar(*args, **styles.sanitize_style(**style))
        self.apply()

        return l

    def imshow(self, ax: plt.Axes, img, extent=None, **kwargs):
        # TODO: default style?
        style = styles.sanitize_style(**kwargs)

        l = ax.imshow(img, extent=extent, **style)
        self.apply(ax)

        return l

    def text(self, ax: plt.Axes, x, y, s: str, **kwargs):
        ax.text(x, y, s, **kwargs)
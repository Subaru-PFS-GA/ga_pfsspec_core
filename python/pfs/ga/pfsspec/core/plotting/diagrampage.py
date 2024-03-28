import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class DiagramPage():
    def __init__(self,
                 nrows, ncols,
                 page_size=None, margins=None, gutters=None,
                 diagram_size=None, dpi=None):
        
        self.__nrows = nrows
        self.__ncols = ncols
        self.__page_size = page_size if page_size is not None else (8, 11)                             # in
        self.__margins = margins if margins is not None else (1, 0.5, 1, 1)             # in
        self.__gutters = gutters if gutters is not None else (0.75, 1.0)
        self.__diagram_size = diagram_size if diagram_size is not None else (3.0, 2.0)
        self.__dpi = dpi if dpi is not None else 240

        self.__f, self.__gs = self.__create_figure()
        self.__ax = {}
        self.__diagrams = []

    def __get_f(self):
        return self.__f
    
    f = property(__get_f)

    def __get_ax(self):
        return self.__ax
    
    ax = property(__get_ax)

    def __get_diagrams(self):
        return self.__diagrams
    
    diagrams = property(__get_diagrams)

    def __create_figure(self):
        # f = plt.figure(figsize=self.__size, dpi=self.__dpi, layout='constrained')
        f = plt.figure(figsize=self.__page_size, dpi=self.__dpi)
        gs = GridSpec(nrows=self.__nrows, ncols=self.__ncols, figure=f, **self.__get_layout())
        return f, gs

    def __get_layout(self):
            # Area within margins
            width = self.__page_size[0] - self.__margins[0] - self.__margins[1]
            height = self.__page_size[1] - self.__margins[2] - self.__margins[3]

            # Columns are justified, rows are aligned to the top

            if self.__ncols == 1:
                width = self.__diagram_size[0]
                wspace = None
            else:
                wspace = max(0, (width - self.__ncols * self.__diagram_size[0]) / (self.__ncols - 1))
                if self.__gutters[0] is not None:
                    wspace = max(wspace, self.__gutters[0])
                # wspace in units of mean diagram size
                wspace /= (width - (self.__ncols - 1) * wspace) / self.__ncols

            if self.__nrows == 1:
                height = self.__diagram_size[1]
                hspace = None
            else:
                height = min(height, self.__nrows * self.__diagram_size[1] + (self.__nrows - 1) * self.__gutters[1])
                # hspace in units of mean diagram size
                hspace = self.__gutters[1] / self.__diagram_size[1]

            left = self.__margins[0] / self.__page_size[0]
            right = (self.__margins[0] + width) / self.__page_size[0]
            top = 1.0 - self.__margins[2] / self.__page_size[1]
            bottom = 1.0 - (self.__margins[2] + height) / self.__page_size[1]

            return dict(
                 left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace
            )
    
    # TODO: delete
    # def __create_axes(self):
    #     # TODO: update this to allow axes spanning gs cells
    #     ax = np.empty((self.__ncols, self.__nrows), dtype=object)
    #     for r in range(self.__nrows):
    #         for c in range(self.__ncols):
    #             ax[c, r] = self.__f.add_subplot(self.__gs[r, c])
    #     return ax
    
    # def __create_diagrams(self):
    #     d = np.empty((self.__ncols, self.__nrows), dtype=object)
    #     return d
    
    def add_diagram(self, key, d):
        if key in self.__ax:
            ax = self.__ax[key]
        else:
            ax = self.__f.add_subplot(self.__gs[key])
            self.__ax[key] = ax

        d._ax = ax
        self.__diagrams.append(d)

        return ax

    def format(self):
        """
        Finalize the format of subplots
        """

        pass

        # for ax in np.atleast_1d(self._ax).ravel():
        #     # ax.legend()
        # self.__f.tight_layout()
    
    def show(self):
        self.__f.show()

    def save(self, file, **kwargs):
        self.__f.savefig(file, **kwargs)
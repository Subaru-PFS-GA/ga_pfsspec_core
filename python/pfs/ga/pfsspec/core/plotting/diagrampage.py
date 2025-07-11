import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

class DiagramPage():
    def __init__(self,
                 *args,
                 title=None,
                 page_size=None, margins=None, gutters=None,
                 diagram_size=None, dpi=None):
        
        if len(args) == 2:
            self.__npages = 1
            self.__nrows = args[0]
            self.__ncols = args[1]
        elif len(args) == 3:
            self.__npages = args[0]
            self.__nrows = args[1]
            self.__ncols = args[2]
        else:
            raise ValueError('Only 2 or 3 positional arguments are allowed')
        
        self.__title = title

        self.__page_size = page_size if page_size is not None else (8, 11)                             # in
        self.__margins = margins if margins is not None else (1, 0.5, 1, 1)             # in
        self.__gutters = gutters if gutters is not None else (0.75, 1.0)
        self.__diagram_size = diagram_size if diagram_size is not None else (3.0, 2.0)
        self.__dpi = dpi if dpi is not None else 240

        self.__ax = {}
        self.__diagrams = []
        self.__f, self.__gs, self.__page_ax = self.__create_figures()

    def __get_title(self):
        return self.__title
    
    def __set_title(self, value):
        self.__title = value

    title = property(__get_title, __set_title)

    def __get_pages(self):
        return self.__f
    
    pages = property(__get_pages)

    def __get_ax(self):
        return self.__ax
    
    ax = property(__get_ax)

    def __get_page_ax(self):
        return self.__page_ax

    page_ax = property(__get_page_ax)

    def __get_diagrams(self):
        return self.__diagrams
    
    diagrams = property(__get_diagrams)

    def __create_figures(self):
        ff = []
        gss = []
        pax = []

        layout = self.__get_layout()

        for p in range(self.__npages):
            f = Figure(figsize=self.__page_size, dpi=self.__dpi)
            gs = GridSpec(nrows=self.__nrows, ncols=self.__ncols, figure=f, **layout)
            ff.append(f)
            gss.append(gs)

            # Create an Axes object that covers the whole page within the margins
            ax = f.add_subplot(gs[:, :])
            ax.set_axis_off()
            pax.append(ax)

            # Set the super title for the page
            if self.__title is not None:
                f.suptitle(self.__title)

        return ff, gss, pax

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
    
    def __parse_key(self, key):
        if len(key) == 2:
            page = 0
        elif len(key) == 3:
            page = key[0]
            key = key[1:]
        else:
            raise ValueError('Key must have 2 or 3 elements.')

        return page, key
    
    def add_axes(self, key):
        page, key = self.__parse_key(key)

        if (page, key) in self.__ax:
            ax = self.__ax[(page, key)]
        else:
            ax = self.__f[page].add_subplot(self.__gs[page][key])
            self.__ax[(page, key)] = ax

        return ax
        
    def add_diagram(self, key, diagram):       
        ax = self.add_axes(key)

        diagram._ax = ax
        self.__diagrams.append(diagram)

        return ax

    def match_limits(self, axs=None):
        if axs is None:
            axs = [ ax for _, ax in self.__ax.items() ]
    
        limits = [ [ None, None ] for _ in range(2) ]

        for ax in axs:
            # Only take into account axes with data and where limits are not set to auto
            if len(ax.lines) > 0 or len(ax.collections) > 0:
                for i, (lim, auto_scale) in enumerate(zip(
                    [ ax.get_xlim(), ax.get_ylim() ],
                    [ax.get_autoscalex_on(), ax.get_autoscaley_on()]
                )):
                    if not auto_scale:
                        for j, stat in enumerate([ min, max ]):
                            if limits[i][j] is None:
                                limits[i][j] = lim[j]
                            elif lim[j] is not None:
                                limits[i][j] = stat(limits[i][j], lim[j])
            
        for ax in axs:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])

    def apply(self):
        pass

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
        if isinstance(file, str):
            dir, filename = os.path.split(file)
            filename, ext = os.path.splitext(filename)

            if ext.lower() == '.pdf':
                # Save to multi-page PDF
                # https://stackoverflow.com/questions/44671169/python-matplotlib-save-to-pdf-in-multiple-pages
                # from https://matplotlib.org/examples/pylab_examples/multipage_pdf.html
                fn = file
                with PdfPages(fn) as pdf:
                    for i in range(self.__npages):
                        pdf.savefig(self.__f[i])
            else:
                for i in range(self.__npages):
                    if self.__npages == 1:
                        fn = file
                    else:
                        fn = file.format(i, page=i)
                        if fn == file:
                            fn = os.path.join(dir, f'{filename}_{i:02d}') + ext
                            
                    self.__f[i].savefig(fn, **kwargs)
        elif self.__npages != 1:
            raise NotImplementedError('Cannot save multi-page diagrams into a stream. Provide a filename as string.')
        else:
            self.__f.savefig(file, **kwargs)
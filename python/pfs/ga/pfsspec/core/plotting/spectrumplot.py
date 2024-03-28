import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .. import Spectrum
from ..util import *
from ..util.args import *
from .diagram import Diagram
from .specaxis import SpecAxis
from .fluxaxis import FluxAxis
from . import styles

class SpectrumPlot(Diagram):
    def __init__(self, ax: plt.Axes = None, diagram_axes=None,
                 title=None,
                 plot_mask=None, plot_flux_err=None, plot_cont=None,
                 orig=None):
        
        if diagram_axes is None:
            diagram_axes = [SpecAxis('nm'), FluxAxis('nJy')]
        
        super().__init__(ax, title=title, diagram_axes=diagram_axes, orig=orig)

        if not isinstance(orig, SpectrumPlot):
            self.plot_mask = plot_mask if plot_mask is not None else False
            self.plot_flux_err = plot_flux_err if plot_flux_err is not None else False
            self.plot_cont = plot_cont if plot_cont is not None else False
        else:
            self.plot_mask = plot_mask if plot_mask is not None else orig.plot_mask
            self.plot_flux_err = plot_flux_err if plot_flux_err is not None else orig.plot_flux_err
            self.plot_cont = plot_cont if plot_cont is not None else orig.plot_cont

        self._validate()

    def _validate(self):
        pass

    def plot_spectrum(self, spectrum, 
                      plot_mask=None, plot_flux_err=None, plot_cont=None,
                      mask_bits=Spectrum.MASK_ANY, 
                      **kwargs):
        
        plot_mask = plot_mask if plot_mask is not None else self.plot_mask,
        plot_flux_err = plot_flux_err if plot_flux_err is not None else self.plot_flux_err
        plot_cont = plot_cont if plot_cont is not None else self.plot_cont

        style = styles.extra_thin_line(**styles.solid_line(**kwargs))

        # First iteration: plot in gray if plotting mask is requested
        # Second iteration: plot unmasked in color
        for i in range(2):
            if i == 0 and plot_mask and spectrum.mask is not None:
                s = styles.lightgray_line(**style)
                m = None
            elif i == 0:
                continue
            else:
                s = style
                if plot_mask and spectrum.mask is not None:
                    # Mask is true when pixel has zero flags set
                    m = spectrum.mask_as_bool(bits=mask_bits)

            m = m if m is not None else np.full_like(spectrum.wave, True, dtype=bool)

            wave, _ = spectrum.wave_in_unit(self.axes[0].unit)
            flux, flux_err = spectrum.flux_in_unit(self.axes[1].unit)

            l = self.plot(self._ax, wave, np.where(m, flux, np.nan), **s)

            # TODO: define styles as use same color as fluxs
            if plot_flux_err and spectrum.flux_err is not None:
                self.plot(self._ax, wave, np.where(m, flux_err, np.nan), **s)

            if plot_cont and spectrum.cont is not None:
                cont = spectrum.cont_in_unit(self.axes[1].unit)
                self.plot(self._ax, wave, np.where(m, cont, np.nan), **s)

        # Set limits
        m &= ~np.isnan(flux) & ~np.isnan(wave)
        if np.sum(m) > 0:
            wmin, wmax = np.quantile(wave, (0.0, 1.0))
            wmin -= (wmax - wmin) * 0.05
            wmax += (wmax - wmin) * 0.05

            fmax = np.quantile(flux[m], 0.95) * 1.1

            # Update min and max
            self.update_limits(0, (wmin, wmax))
            self.update_limits(1, (0, fmax))

        self.apply()

        return l
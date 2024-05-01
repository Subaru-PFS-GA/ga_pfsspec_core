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

    def get_limits(self, wave, flux, mask=None, plot_residual=False):
        mask = mask if mask is not None else np.full_like(flux, True, dtype=bool)
        mask &= ~np.isnan(flux) & ~np.isnan(wave)
        
        if np.sum(mask) > 0:
            wmin, wmax = np.quantile(wave[~np.isnan(wave)], (0.0, 1.0))
            wmin -= (wmax - wmin) * 0.05
            wmax += (wmax - wmin) * 0.05

            if not plot_residual:
                # TODO: This works well with noisy spectra but what if we
                #       plot a template?
                fmax = np.quantile(flux[mask], 0.95) * 1.1
            else:
                fmax = np.quantile(flux[mask], 0.95) * 1.1
        else:
            wmin = None
            wmax = None
            fmax = None

        return wmin, wmax, fmax
    
    def _plot_spectrum(self, wave, flux, flux_err, cont, mask, style,
                       plot_mask=None, plot_flux_err=None, plot_cont=None,
                       plot_residual=False):
        
        plot_mask = plot_mask if plot_mask is not None else self.plot_mask,
        plot_flux_err = plot_flux_err if plot_flux_err is not None else self.plot_flux_err
        plot_cont = plot_cont if plot_cont is not None else self.plot_cont

        # First iteration: plot in gray if plotting mask is requested
        # Second iteration: plot unmasked in color
        for i in range(2):
            if i == 0 and plot_mask and mask is not None:
                s = styles.lightgray_line(**style)
                m = None
            elif i == 0:
                continue
            else:
                s = style
                if plot_mask and mask is not None:
                    # Mask is true when pixel has zero flags set
                    m = mask

            m = mask if mask is not None else np.full_like(wave, True, dtype=bool)

            l = self.plot(self._ax, wave, np.where(m, flux, np.nan), **s)

            # TODO: define styles as use same color as fluxs
            if plot_flux_err and flux_err is not None:
                self.plot(self._ax, wave, np.where(m, flux_err, np.nan), **s)

            if plot_cont and cont is not None:
                self.plot(self._ax, wave, np.where(m, cont, np.nan), **s)

        # Set limits
        if np.sum(m) > 0:
            wmin, wmax, fmax = self.get_limits(wave, flux, mask=m, plot_residual=plot_residual)

            # Update min and max
            self.update_limits(0, (wmin, wmax))
            if not plot_residual:
                self.update_limits(1, (0, fmax))
            else:
                self.update_limits(1, (-fmax, fmax))

        self.apply()

        return l

    def plot_spectrum(self, spectrum, 
                      plot_mask=None, plot_flux_err=None, plot_cont=None,
                      mask_bits=Spectrum.MASK_ANY, 
                      **kwargs):
        
        style = styles.extra_thin_line(**styles.solid_line(**kwargs))

        mask = spectrum.mask_as_bool(bits=mask_bits)
        wave, _ = spectrum.wave_in_unit(self.axes[0].unit)
        flux, flux_err = spectrum.flux_in_unit(self.axes[1].unit)
        if spectrum.cont is not None:
            cont = spectrum.cont_in_unit(self.axes[1].unit)
        else:
            cont = None

        l = self._plot_spectrum(wave, flux, flux_err, cont, mask, style,
                                plot_mask=plot_mask,
                                plot_flux_err=plot_flux_err,
                                plot_cont=plot_cont)

        return l
    
    def plot_template(self, template, **kwargs):
        # Plot a spectrum template

        style = styles.extra_thin_line(**styles.solid_line(**kwargs))
        style = styles.red_line(**style)

        wave, _ = template.wave_in_unit(self.axes[0].unit)
        flux, flux_err = template.flux_in_unit(self.axes[1].unit)

        l = self._plot_spectrum(wave, flux, flux_err, None, None, style)

        return l
    
    def plot_residual(self, spectrum, template,
                      plot_mask=None, plot_flux_err=None, plot_cont=None,
                      mask_bits=Spectrum.MASK_ANY, 
                      **kwargs):
        
        style = styles.extra_thin_line(**styles.solid_line(**kwargs))

        mask = spectrum.mask_as_bool(bits=mask_bits)
        wave, _ = spectrum.wave_in_unit(self.axes[0].unit)
        flux, _ = spectrum.flux_in_unit(self.axes[1].unit)
        template_flux, _ = template.flux_in_unit(self.axes[1].unit)

        flux = flux - template_flux

        l = self._plot_spectrum(wave, flux, None, None, mask, style,
                                plot_mask=plot_mask,
                                plot_flux_err=plot_flux_err,
                                plot_cont=False,
                                plot_residual=True)

        return l


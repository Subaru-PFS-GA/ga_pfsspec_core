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

        self._mask_ax = None

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
    
    def _get_wave_mask(self, spectrum, wlim):
        if wlim is not None:
            wnm, _ = spectrum.wave_in_unit('nm')
            wave_mask = (wnm >= wlim[0]) & (wnm <= wlim[1])
        else:
            wave_mask = np.full_like(spectrum.wave, True, dtype=bool)

        return wave_mask
    
    def _plot_spectrum(self, wave, flux, flux_err, cont, mask, style,
                       mask_bits=None, mask_flags=None, s=None,
                       plot_mask=None, plot_flux_err=None, plot_cont=None,
                       plot_residual=False,
                       auto_limits=False):
        
        plot_mask = plot_mask if plot_mask is not None else self.plot_mask,
        plot_flux_err = plot_flux_err if plot_flux_err is not None else self.plot_flux_err
        plot_cont = plot_cont if plot_cont is not None else self.plot_cont

        def apply_slice(vector):
            if s is not None:
                return vector[s] if vector is not None else None
            else:
                return vector
            
        def safe_plot(wave, flux, m, ax=None, **kwargs):
            # Only plot if there are unmasked pixels, otherwise plot limits
            # would be set to weird values

            ax = ax if ax is not None else self._ax

            x = apply_slice(wave)
            if m is not None:
                y = apply_slice(np.where(m, flux, np.nan))
            else:
                y = apply_slice(flux)

            if np.any(~np.isnan(x) & ~np.isinf(x) & ~np.isnan(y) & ~np.isinf(y)):
                return self.plot(ax, x, y, **kwargs)
            else: 
                return None
            
        def get_mask():
            if mask is not None:
                # Mask is true when pixel has zero flags set
                if mask is not None:
                    if mask.dtype == bool:
                        m = mask
                    elif mask_bits is not None:
                        m = (mask & mask_bits) == 0
                    else:
                        m = mask == 0
            else:
                m = None

            return m
            
        if np.size(apply_slice(wave)) == 0:
            return None
        
        Z_ORDER_MASKED_FLUX = 2.0
        Z_ORDER_FLUX_ERR = 2.1
        Z_ORDER_FLUX = 2.5
        Z_ORDER_CONT = 2.6
        Z_ORDER_MASK = 2.7 
    
        # Plot flux
        # First iteration: plot everything in gray if plotting mask is requested
        # Second iteration: plot unmasked pixels in color
        for i in range(2):
            if i == 0 and mask is not None:
                ss = styles.lightgray_line(**style)
                zorder = Z_ORDER_MASKED_FLUX
                m = None
            elif i == 0:
                # No mask, continue with plotting unmasked spectrum
                continue
            elif i == 1:
                ss = style
                zorder = Z_ORDER_FLUX
                m = get_mask()
            else:
                raise NotImplementedError()

            l = safe_plot(wave, flux, m, zorder=zorder, **ss)

        # Plot flux error
        if plot_flux_err and flux_err is not None:
            safe_plot(wave, flux_err, None, zorder=Z_ORDER_FLUX_ERR, **styles.lightblue_line(**style))

        # Plot continuum
        if plot_cont and cont is not None:
            safe_plot(wave, cont, None, zorder=Z_ORDER_CONT **styles.red_line(**style))

        # Plot the mask bits
        if plot_mask and mask is not None:
            # Create a secondary axis for the mask on the right
            ax, r = self.__get_mask_axis(mask, mask_flags)
    
            for i in range(r):
                mm = np.where(mask & (1 << i), i, np.nan)
                self.plot(ax, wave, mm, zorder=Z_ORDER_MASK, **styles.red_line(**styles.solid_line(**ss)))

        # Set limits
        if auto_limits:
            if np.sum(m) == 0:
                # The entire spectrum is masked
                self.update_limits(0, (None, None))
                self.update_limits(1, (None, None))
            else:
                m = get_mask()
                wmin, wmax, fmax = self.get_limits(
                    apply_slice(wave),
                    apply_slice(flux),
                    mask=apply_slice(m),
                    plot_residual=plot_residual)

                # Update min and max
                self.update_limits(0, (wmin, wmax))
                if not plot_residual:
                    self.update_limits(1, (0, fmax))
                else:
                    self.update_limits(1, (-fmax, fmax))

        self.apply()

        return l
    
    def __get_mask_axis(self, mask, mask_flags):
        if mask.dtype == bool:
            r = 1
        elif mask.dtype == np.int32 or mask.dtype == np.uint32:
            r = 31
        elif mask.dtype == np.int64 or mask.dtype == np.uint64:
            r = 63
        else:
            raise NotImplementedError()
        
        if self._mask_ax is None:
            self._mask_ax = self._ax.twinx()
            self._mask_ax.set_ylim(-1, r + 1)

            # Add labels to the axis if mask_flags is provided
            ticks = []
            labels = []
            if mask_flags is not None:
                for i in range(r):
                    if i in mask_flags:
                        ticks.append(i)
                        labels.append(mask_flags[i])
                self._mask_ax.set_yticks(ticks, labels)
                self._mask_ax.tick_params(axis='y', which='both', **styles.mask_label_font())

        return self._mask_ax, r


    def plot_spectrum(self, spectrum, 
                      plot_mask=None, plot_flux_err=None, plot_cont=None,
                      flux_corr=None,
                      mask_bits=None,
                      wlim=None,
                      auto_limits=False,
                      **kwargs):
        
        style = styles.extra_thin_line(**styles.solid_line(**kwargs))

        mask = spectrum.mask
        mask_flags = spectrum.mask_flags
        wave, _ = spectrum.wave_in_unit(self.axes[0].unit)
        flux, flux_err = spectrum.flux_in_unit(self.axes[1].unit)

        if flux_corr is not None:
            flux = flux / flux_corr
            if flux_err is not None:
                flux_err = flux_err / flux_corr

        if spectrum.cont is not None:
            cont = spectrum.cont_in_unit(self.axes[1].unit)
        else:
            cont = None

        wave_mask = self._get_wave_mask(spectrum, wlim)
        l = self._plot_spectrum(wave,
                                flux,
                                flux_err,
                                cont,
                                mask,
                                style=style,
                                mask_bits=mask_bits,
                                mask_flags=mask_flags,
                                s=wave_mask,
                                plot_mask=plot_mask,
                                plot_flux_err=plot_flux_err,
                                plot_cont=plot_cont,
                                auto_limits=auto_limits)
        return l
    
    def plot_template(self, template, wlim=None, auto_limits=False, **kwargs):
        # Plot a spectrum template

        style = styles.extra_thin_line(**styles.solid_line(**kwargs))
        style = styles.red_line(**style)

        wave, _ = template.wave_in_unit(self.axes[0].unit)
        flux, flux_err = template.flux_in_unit(self.axes[1].unit)

        wave_mask = self._get_wave_mask(template, wlim)
        l = self._plot_spectrum(wave, flux, flux_err, None, None, style, s=wave_mask, auto_limits=auto_limits)
        return l
    
    def plot_residual(self, spectrum, template,
                      plot_mask=None, plot_flux_err=None, plot_cont=None,
                      mask_bits=None, 
                      wlim=None,
                      auto_limits=False,
                      **kwargs):
        
        style = styles.extra_thin_line(**styles.solid_line(**kwargs))

        mask = spectrum.mask_as_bool(bits=mask_bits)
        wave, _ = spectrum.wave_in_unit(self.axes[0].unit)
        flux, _ = spectrum.flux_in_unit(self.axes[1].unit)
        template_flux, _ = template.flux_in_unit(self.axes[1].unit)

        flux = flux - template_flux

        wave_mask = self._get_wave_mask(spectrum, wlim)
        l = self._plot_spectrum(wave, flux, None, None, mask,
                                style,
                                s=wave_mask,
                                plot_mask=plot_mask,
                                plot_flux_err=plot_flux_err,
                                plot_cont=False,
                                plot_residual=True,
                                auto_limits=auto_limits)
        return l


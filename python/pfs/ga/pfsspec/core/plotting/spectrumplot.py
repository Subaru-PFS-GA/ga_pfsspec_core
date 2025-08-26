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

    Z_ORDER_MASKED_FLUX = 2.0
    Z_ORDER_FLUX_ERR = 2.1
    Z_ORDER_FLUX = 2.5
    Z_ORDER_CONT = 2.6
    Z_ORDER_MASK = 2.7
    Z_ORDER_SHADE = 2.8

    def __init__(self, ax: plt.Axes = None, diagram_axes=None,
                 title=None,
                 plot_mask=None, plot_flux=None, plot_flux_err=None, plot_cont=None, plot_model=None,
                 print_snr=None,
                 flux_calibrated=True,
                 orig=None):
        
        if diagram_axes is None:
            if flux_calibrated:
                diagram_axes = [SpecAxis('nm'), FluxAxis('nJy')]
            else:
                diagram_axes = [SpecAxis('nm'), FluxAxis('ADU')]
        
        super().__init__(ax, title=title, diagram_axes=diagram_axes, orig=orig)

        if not isinstance(orig, SpectrumPlot):
            self.plot_mask = plot_mask if plot_mask is not None else False
            self.plot_flux = plot_flux if plot_flux is not None else True
            self.plot_flux_err = plot_flux_err if plot_flux_err is not None else False
            self.plot_cont = plot_cont if plot_cont is not None else False
            self.plot_model = plot_model if plot_model is not None else False
            self.print_snr = print_snr if print_snr is not None else False

            self.auto_limits_mask_min = 10              # Minimum number of unmasked pixels
            self.auto_limits_flux_quantile = 0.95
            self.auto_limits_error_max = 0.5
        else:
            self.plot_mask = plot_mask if plot_mask is not None else orig.plot_mask
            self.plot_flux = plot_flux if plot_flux is not None else orig.plot_flux
            self.plot_flux_err = plot_flux_err if plot_flux_err is not None else orig.plot_flux_err
            self.plot_cont = plot_cont if plot_cont is not None else orig.plot_cont
            self.plot_model = plot_model if plot_model is not None else orig.plot_model
            self.print_snr = print_snr if print_snr is not None else orig.print_snr

            self.auto_limits_mask_min = orig.auto_limits_mask_min
            self.auto_limits_flux_quantile = orig.auto_limits_flux_quantile
            self.auto_limits_error_max = orig.auto_limits_error_max

        self._mask_ax = None

        self._validate()

    def _validate(self):
        pass

    def get_limits(self, wave, flux, /, flux_err=None, mask=None, plot_residual=False):
        mask = mask if mask is not None else np.full_like(flux, True, dtype=bool)
        mask &= ~np.isnan(flux) & ~np.isnan(wave)
        if flux_err is not None:
            mask &= (flux_err / flux < self.auto_limits_error_max)
        
        if np.sum(mask) >= self.auto_limits_mask_min:
            wmin, wmax = np.quantile(wave[~np.isnan(wave)], (0.0, 1.0))
            offset = (wmax - wmin) * 0.05
            wmin -= offset
            wmax += offset

            q = np.quantile(flux[mask], self.auto_limits_flux_quantile)
            offset = 2 * (1 - self.auto_limits_flux_quantile)

            if not plot_residual:
                # TODO: This works well with noisy spectra but what if we
                #       plot a template?
                fmin = -offset * q
                fmax = (1 + offset) * q
            else:
                fmin = -(1 + offset) * q
                fmax = (1 + offset) * q
        else:
            wmin = None
            wmax = None
            fmin = None
            fmax = None

        return wmin, wmax, fmin, fmax
    
    def _get_wave_mask(self, spectrum, wlim):
        if wlim is not None:
            wnm, _ = spectrum.wave_in_unit('nm')
            wave_mask = (wnm >= wlim[0]) & (wnm <= wlim[1])
        else:
            wave_mask = np.full_like(spectrum.wave, True, dtype=bool)

        return wave_mask
    
    def _plot_spectrum(self, wave, flux, flux_err, cont, model, mask, /, 
                       style={},
                       mask_bits=None, mask_flags=None,
                       snr=None,
                       plot_flux=None, plot_flux_err=None, plot_cont=None, plot_model=None,
                       plot_residual=False,
                       plot_mask=None, plot_nan=None,
                       print_snr=None,
                       s=None,
                       auto_limits=False):

        """
        Plot a spectrum, an optionally the flux error, continuum or mask.

        Parameters
        ----------
        wave : numpy.ndarray
            Wavelength array.
        flux : numpy.ndarray
            Flux array.
        flux_err : numpy.ndarray
            Flux error array.
        cont : numpy.ndarray
            Continuum array.
        model : numpy.ndarray
            Best fit model array.
        mask : numpy.ndarray
            Mask array. Boolean or integer.
        style : dict
            Style dictionary.
        mask_bits : int
            Mask bits to treat as masked pixels. If None, all non-zero values are
            treated as masked.
        plot_flux : bool
            Plot the flux vector.
        plot_flux_err : bool
            Plot the flux error vector.
        plot_cont : bool
            Plot the continuum vector.
        plot_residual : bool
            Assume that the flux vector is a residual. Limits of the y-axis will
            be set symmetrically around zero.
        plot_mask : bool
            Plot the mask bits.
        plot_nan : bool
            Plot a dot where NaN values are present.
        s : slice
            Slice to apply to the vectors.
        auto_limits : bool
            Automatically set limits based on the data.
        """
        
        plot_flux = plot_flux if plot_flux is not None else self.plot_flux,
        plot_flux_err = plot_flux_err if plot_flux_err is not None else self.plot_flux_err
        plot_cont = plot_cont if plot_cont is not None else self.plot_cont
        plot_model = plot_model if plot_model is not None else self.plot_model
        plot_mask = plot_mask if plot_mask is not None else self.plot_mask
        plot_nan = plot_nan if plot_nan is not None else False

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

        m = None
        l = None
    
        if plot_flux and flux is not None:
            # First iteration: plot everything in gray if plotting mask is requested
            # Second iteration: plot unmasked pixels in color
            for i in range(2):
                if i == 0 and mask is not None:
                    ss = styles.lightgray_line(**style)
                    zorder = SpectrumPlot.Z_ORDER_MASKED_FLUX
                    m = None
                elif i == 0:
                    # No mask, continue with plotting unmasked spectrum
                    continue
                elif i == 1:
                    ss = style
                    zorder = SpectrumPlot.Z_ORDER_FLUX
                    m = get_mask()
                else:
                    raise NotImplementedError()

                l = safe_plot(wave,
                              flux,
                              m,
                              zorder=zorder,
                              **styles.alpha_line(**ss))

        # Plot flux error
        if plot_flux_err and flux_err is not None:
            l2 = safe_plot(wave,
                           flux_err,
                           None,
                           zorder=SpectrumPlot.Z_ORDER_FLUX_ERR,
                           **styles.alpha_line(**styles.blue_line(**style)))
            if l is None:
                l = l2

        # Plot continuum
        if plot_cont and cont is not None:
            l2 = safe_plot(wave, cont, None, zorder=SpectrumPlot.Z_ORDER_CONT, **styles.red_line(**style))
            if l is None:
                l = l2

        # Plot model
        if plot_model and model is not None:
            l2 = safe_plot(wave, model, None, zorder=SpectrumPlot.Z_ORDER_CONT, **styles.red_line(**style))
            if l is None:
                l = l2

        if plot_mask or plot_nan:
            # Create a secondary axis for the mask on the right
            mask_ax, mask_r = self.__get_mask_axis(mask, mask_flags, plot_nan)

        # Plot the mask bits
        if plot_mask and mask is not None:
            for i in range(mask_r + 1):
                mm = np.where(mask & (1 << i), i, np.nan)
                if plot_nan:
                    mm += 1
                self.plot(mask_ax, wave, mm, zorder=SpectrumPlot.Z_ORDER_MASK, **styles.red_line(**styles.solid_line(**ss)))

        if plot_nan and flux is not None:
            # TODO: what if we don't plot the mask?)
            mm = np.where(np.isnan(flux) | np.isinf(flux), 0, np.nan)
            self.plot(mask_ax, wave, mm, zorder=SpectrumPlot.Z_ORDER_MASK, **styles.blue_line(**styles.solid_line(**ss)))

        # Calculate the limits
        m = get_mask()
        f = flux if flux is not None else cont
        e = flux_err

        if np.sum(m) > 0:
            wmin, wmax, fmin, fmax = self.get_limits(
                apply_slice(wave),
                apply_slice(f),
                flux_err=apply_slice(e),
                mask=apply_slice(m),
                plot_residual=plot_residual)            
        else:
            wmin, wmax, fmin, fmax = None, None, None, None

        # Print the SNR
        if print_snr and snr is not None:
            self.text(self._ax, wave.mean(), fmax, f'S/N = {snr:0.2f}')

        # Set limits
        if auto_limits:
            if np.sum(m) == 0:
                # The entire spectrum is masked
                self.update_limits(0, (None, None))
                self.update_limits(1, (None, None))
            else:
                # Update min and max
                self.update_limits(0, (wmin, wmax))
                self.update_limits(1, (fmin, fmax))

        self.apply()

        return l
    
    def __get_mask_axis(self, mask, mask_flags, include_nan=False):
        """
        Create a right axis to display mask bits with labels.

        Parameters
        ----------
        mask : numpy.ndarray
            Mask array. Its dtype is used to determine the maximum number of bits.
        mask_flags : dict
            Dictionary with mask bit labels.
        include_nan : bool
            Include a label for NaN and INF values.
        """

        r = 0
        if mask.dtype == bool:
            r = 1
        elif mask.dtype == np.int32 or mask.dtype == np.uint32 or \
             mask.dtype == np.int64 or mask.dtype == np.uint64:
            
            # Find most significant bit of the mask
            allbits = np.bitwise_or.reduce(mask.astype(np.uint64))
            for i in range(64, 0, -1):
                if (allbits & np.uint64(1 << (i - 1))) != 0:
                    r = i - 1
                    break

            # Find the most significant bit in mask_flags
            if mask_flags is not None:
                for i in mask_flags:
                    if i > r:
                        r = i
        else:
            raise NotImplementedError()
        
        if self._mask_ax is None:
            self._mask_ax = self._ax.twinx()

            ymax = r + 1 if include_nan else r
            self._mask_ax.set_ylim(-1, ymax + 1)

            # Add labels to the axis if mask_flags is provided
            ticks = []
            labels = []

            if include_nan:
                ticks.append(0)
                labels.append('NAN or INF')

            if mask_flags is not None:
                for i in range(r + 1):
                    if i in mask_flags:
                        ticks.append(i + 1 if include_nan else i)
                        labels.append(mask_flags[i])

            if len(ticks) > 0:
                self._mask_ax.set_yticks(ticks, labels)
                self._mask_ax.tick_params(axis='y', which='both', **styles.mask_label_font())

        return self._mask_ax, r

    def plot_spectrum(self, spectrum, /,
                      plot_flux=None, plot_flux_err=None, plot_cont=None, plot_model=None,
                      plot_mask=None, plot_nan=None,
                      print_snr=False,
                      apply_flux_corr=False,
                      normalize_cont=False,
                      mask_bits=None,
                      wlim=None,
                      auto_limits=False,
                      **kwargs):
        
        """
        Plot a spectrum.

        Parameters
        ----------
        spectrum : Spectrum
            A spectrum object to plot with at least the wave and flux attributes set.
        plot_flux : bool
            Plot flux.
        plot_flux_err : bool
            Plot flux error, if available.
        plot_cont : bool
            Plot continuum, if available.
        plot_model : bool
            Plot best fit model, if available.
        plot_mask : bool
            Plot mask bits.
        plot_nan : bool
            Plot a dot where NaN values are present.
        mask_bits : int
            Bits to treat as a masked pixel when plotting the spectrum.
        wlim : tuple
            Wavelength limits of the plot.
        auto_limits : bool
            Automatically set limits based on the data.
        kwargs : dict
            These are passed to the matplotlib plotting functions.
        """
        
        style = styles.extra_thin_line(**styles.solid_line(**kwargs))

        mask = spectrum.mask
        mask_flags = spectrum.mask_flags
        mask_bits = mask_bits if mask_bits is not None else spectrum.mask_bits
        wave, _ = spectrum.wave_in_unit(self.axes[0].unit)

        # By default, assume a calibrated spectrum
        if spectrum.is_flux_calibrated is not None and not spectrum.is_flux_calibrated:
            flux, flux_err = spectrum.flux, spectrum.flux_err
        else:
            flux, flux_err = spectrum.flux_in_unit(self.axes[1].unit)

        if spectrum.cont is not None:
            if spectrum.is_flux_calibrated is not None and not spectrum.is_flux_calibrated:
                cont = spectrum.cont
            else:
                cont = spectrum.cont_in_unit(self.axes[1].unit)
        else:
            cont = None

        if spectrum.flux_model is not None:
            if spectrum.is_flux_calibrated is not None and not spectrum.is_flux_calibrated:
                model = spectrum.flux_model
            else:
                model = spectrum.model_in_unit(self.axes[1].unit)
        else:
            model = None

        # Flux correction is a unitless, wavelength-dependent factor
        if apply_flux_corr and spectrum.flux_corr is not None:
            flux = flux * spectrum.flux_corr
            flux_err = flux_err * spectrum.flux_corr if flux_err is not None else None

        # Normalize with the continuum, the continuum has a unit
        if normalize_cont and cont is not None:
            flux = flux / cont
            flux_err = flux_err / cont if flux_err is not None else None

        wave_mask = self._get_wave_mask(spectrum, wlim)
        l = self._plot_spectrum(wave,
                                flux,
                                flux_err,
                                cont,
                                model,
                                mask,
                                snr=spectrum.snr,
                                style=style,
                                mask_bits=mask_bits,
                                mask_flags=mask_flags,
                                s=wave_mask,
                                plot_flux=plot_flux,
                                plot_flux_err=plot_flux_err,
                                plot_cont=plot_cont,
                                plot_mask=plot_mask,
                                plot_nan=plot_nan,
                                print_snr=print_snr,
                                auto_limits=auto_limits)
        return l
    
    def plot_template(self, template, /,
                      plot_flux=None, plot_cont=None,
                      plot_mask=None, plot_nan=None,
                      apply_flux_corr=False,
                      normalize_cont=False,
                      mask_bits=None,
                      wlim=None,
                      auto_limits=False,
                      **kwargs):
        
        """
        Plot a spectrum template. This is similar to plot_spectrum but it assumes
        that the spectrum is a template so flux correction and continuum normalization
        is applied differently and no flux error is assumed.
        """

        style = styles.extra_thin_line(**styles.solid_line(**kwargs))
        style = styles.red_line(**style)

        mask = template.mask
        mask_flags = template.mask_flags
        mask_bits = mask_bits if mask_bits is not None else template.mask_bits
        wave, _ = template.wave_in_unit(self.axes[0].unit)

        if template.is_flux_calibrated is not None and not template.is_flux_calibrated:
            flux, flux_err = template.flux, template.flux_err
        else:
            flux, flux_err = template.flux_in_unit(self.axes[1].unit)

        if template.cont is not None:
            if template.is_flux_calibrated is not None and not template.is_flux_calibrated:
                cont = template.cont
            else:
                cont = template.cont_in_unit(self.axes[1].unit)
        else:
            cont = None

        # Flux correction is a unitless, wavelength-dependent factor
        if apply_flux_corr and template.flux_corr is not None:
            flux = flux * template.flux_corr
            flux_err = flux_err * template.flux_corr if flux_err is not None else None

        # Normalize with the continuum, the continuum has a unit
        if normalize_cont and cont is not None:
            flux = flux / cont
            flux_err = flux_err / cont if flux_err is not None else None

        wave_mask = self._get_wave_mask(template, wlim)
        l = self._plot_spectrum(wave,
                                flux,
                                flux_err,
                                cont,
                                None,
                                mask,
                                style=style,
                                mask_bits=mask_bits,
                                mask_flags=mask_flags,
                                s=wave_mask,
                                plot_flux=plot_flux,
                                plot_cont=plot_cont,
                                plot_mask=plot_mask,
                                plot_nan=plot_nan,
                                auto_limits=auto_limits)
        return l
    
    def plot_residual(self, spectrum, template, /,
                      plot_flux=None,
                      plot_mask=None, plot_nan=None,
                      mask_bits=None,
                      wlim=None,
                      auto_limits=False,
                      **kwargs):
        
        style = styles.extra_thin_line(**styles.solid_line(**kwargs))

        mask = spectrum.mask_as_bool(bits=mask_bits) & template.mask_as_bool(bits=mask_bits)
        wave, _ = spectrum.wave_in_unit(self.axes[0].unit)

        if spectrum.is_flux_calibrated is not None and not spectrum.is_flux_calibrated:
            flux = spectrum.flux
        else:
            flux, _ = spectrum.flux_in_unit(self.axes[1].unit)

        if template.is_flux_calibrated is not None and not template.is_flux_calibrated:
            template_flux = template.flux
        else:
            template_flux, _ = template.flux_in_unit(self.axes[1].unit)

        # The residual is the difference between the spectrum and the template
        flux = flux - template_flux

        wave_mask = self._get_wave_mask(spectrum, wlim) & self._get_wave_mask(template, wlim)
        l = self._plot_spectrum(wave,
                                flux,
                                None,
                                None,
                                None,
                                mask,
                                style=style,
                                s=wave_mask,
                                plot_flux=True,
                                plot_flux_err=False,
                                plot_cont=False,
                                plot_mask=plot_mask,
                                plot_nan=plot_nan,
                                plot_residual=True,
                                auto_limits=auto_limits)
        return l
    
    def shade_wave_ranges(self, ranges, **kwargs):
        # Shade the masked regions

        style = styles.red_shade(**kwargs)

        if ranges is not None:
            for (wmin, wmax) in ranges:
                wmin = Spectrum._wave_in_unit(wmin, self.axes[0].unit)
                wmax = Spectrum._wave_in_unit(wmax, self.axes[0].unit)
                self._ax.axvspan(wmin, wmax, zorder=SpectrumPlot.Z_ORDER_SHADE, **style)
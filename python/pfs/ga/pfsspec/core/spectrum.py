import logging
import numpy as np
import numbers
import collections
import matplotlib.pyplot as plt

try:
    import pysynphot
    import pysynphot.binning
    import pysynphot.spectrum
    import pysynphot.reddening
except ModuleNotFoundError as ex:
    logging.warning(ex.msg)
    pysynphot = None

from pfs.ga.pfsspec.core.util.copy import *
from pfs.ga.pfsspec.core.pfsobject import PfsObject
from pfs.ga.pfsspec.core.obsmod.psf import *
from .physics import Physics
from .constants import Constants

class Spectrum(PfsObject):

    MASK_DEFAULT = 0x1                      # Default lowest bit mask
    MASK_NOWAVE = 0x10000                   # Wavelength of pixel is not calibrated
    MASK_NOSKY = 0x20000                    # No sky model available
    MASK_SKYLINE = 0x30000                  # Strong sky line
    MASK_EXTRAPOLATED = 0x40000             # Values are extrapolated
    MASK_ANY = 0xEFFFFFFF                   # All bits (signed int)


    # PFS masks:

    # BAD: the pixel is a known bad pixel
    # SAT: the pixel was saturated
    # INTRP: the pixel has been interpolated
    # CR: the pixel is believed to be affected by a cosmic ray
    # EDGE: the pixel is close to the edge of the image (from LSST; not relevant for PFS)
    # DETECTED: the pixel has a positive value over threshold (from LSST; not relevant for PFS)
    # DETECTED_NEGATIVE: the pixel has a negative value over threshold (from LSST; not relevant for PFS)
    # SUSPECT: the pixel is suspected to be non-linear
    # NO_DATA: there are no good value for this value
    # UNMASKEDNAN: the pixel contained a NAN
    # BAD_FLAT: the pixel is bad in the flat
    # FIBERTRACE: the pixel is part of a fiber trace
    # BAD_SKY: the pixel is bad in the sky model
    # BAD_FLUXCAL: the pixel is bad in the flux calibration
    # OVERLAP: the pixel is part of a wavelength overlap (dichroic)




    def __init__(self, orig=None):
        super().__init__(orig=orig)
        
        if not isinstance(orig, Spectrum):
            self.index = None
            self.id = 0

            self.redshift = 0.0
            self.redshift_err = 0.0

            # TODO: these should go elsewhere, they apply to simulated spectra only
            self.exp_count = None
            self.exp_time = None
            self.seeing = None
            self.ext = None
            self.target_zenith_angle = None
            self.target_field_angle = None
            self.moon_zenith_angle = None
            self.moon_target_angle = None
            self.moon_phase = None
            self.snr = None
            self.mag = None
            ###

            self.spectrograph = None
            self.fiberid = None

            self.wave = None
            self.wave_edges = None
            self.flux = None                    # Flux (+noise when frozen)
            self.flux_model = None              # Flux (model, when noise is frozen)
            self.flux_err = None                # Flux error (sigma, not squared)
            self.flux_sky = None                # Background flux (sky + moon, no detector)
            self.flux_calibration = None
            self.mask = None
            self.weight = None
            self.cont = None
            self.cont_fit = None

            self.resolution = None
            self.is_wave_regular = None
            self.is_wave_lin = None
            self.is_wave_log = None

            self.aux_params = {}
            self.history = []
        else:
            self.index = orig.index
            self.id = orig.id

            # TODO: move these from the base class
            self.redshift = orig.redshift
            self.redshift_err = orig.redshift_err
            self.exp_count = orig.exp_count
            self.exp_time = orig.exp_time
            self.seeing = orig.seeing
            self.ext = orig.ext
            self.target_zenith_angle = orig.target_zenith_angle
            self.target_field_angle = orig.target_field_angle
            self.moon_zenith_angle = orig.moon_zenith_angle
            self.moon_target_angle = orig.moon_target_angle
            self.moon_phase = orig.moon_phase
            self.snr = orig.snr
            self.mag = orig.mag
            ###

            self.fiberid = safe_deep_copy(orig.fiberid)
            
            self.wave = safe_deep_copy(orig.wave)
            self.wave_edges = safe_deep_copy(orig.wave_edges)
            self.flux = safe_deep_copy(orig.flux)
            self.flux_model = safe_deep_copy(orig.flux_model)
            self.flux_err = safe_deep_copy(orig.flux_err)
            self.flux_sky = safe_deep_copy(orig.flux_sky)
            self.flux_calibration = safe_deep_copy(orig.flux_calibration)
            self.mask = safe_deep_copy(orig.mask)
            self.weight = safe_deep_copy(orig.weight)
            self.cont = safe_deep_copy(orig.cont)
            self.cont_fit = orig.cont_fit

            self.resolution = orig.resolution
            self.is_wave_regular = orig.is_wave_regular
            self.is_wave_lin = orig.is_wave_lin
            self.is_wave_log = orig.is_wave_log

            self.aux_params = safe_deep_copy(orig.aux_params)
            self.history = safe_deep_copy(orig.history)

    def copy(self):
        s = super().copy()
        s.append_history('A copy of a spectrum is created.')
        return s

    def append_history(self, msg):
        self.history.append(msg)

    def get_param_names(self):
        return ['id',
                'redshift',
                'redshift_err',
                'exp_count',
                'exp_time',
                'seeing',
                'ext',
                'target_zenith_angle',
                'target_field_angle',
                'moon_zenith_angle',
                'moon_target_angle',
                'moon_phase',
                'snr',
                'mag',
                'fiberid',
                'cont_fit',
                'random_seed'] + list(self.aux_params.keys())

    def get_params(self):
        params = {}
        for p in self.get_param_names():
            params[p] = getattr(self, p)
        for p, v in self.aux_params.items():
            params[p] = self.aux_params[p]
        return params

    def set_params(self, params):
        for p in self.get_param_names():
            if p in params:
                setattr(self, p, params[p])

    def get_params_as_datarow(self):
        row = {}
        for p in self.get_param_names():
            if p in self.aux_params:
                v = self.aux_params[p]
            else:
                v = getattr(self, p)

            # If parameter is an array, assume it's equal length and copy to
            # pandas as a set of columns instead of a single column
            if v is None:
                row[p] = np.nan
            elif isinstance(v, np.ndarray):
                if len(v.shape) > 1:
                    raise Exception('Can only serialize one-dimensional arrays')
                for i in range(v.shape[0]):
                    row['{}_{}'.format(p, i)] = v[i]
            else:
                row[p] = v

        return row
    
    #region Mask manipulation
    
    def merge_mask(self, mask, bits=MASK_DEFAULT):
        self.mask = self.get_mask_merged(self.wave, self.mask, mask, bits2=bits)

    def mask_as_bool(self, bits=MASK_ANY):
        return self.get_mask_as_bool(self.wave, self.mask, bits=bits)
    
    def mask_as_int(self):
        return self.get_mask_as_int(self.wave, self.mask)
    
    def mask_from_wave(self):
        return ~np.isnan(self.wave)

    @staticmethod
    def get_mask_as_bool(wave, mask, bits=MASK_ANY):
        """
        Return array which is True if the pixel is not maskeds.
        """
        if mask is None:
            return None
        elif isinstance(mask, slice):
            m = np.full_like(wave, False)
            m[mask] = True
            return m
        elif isinstance(mask, np.ndarray) and mask.dtype == bool:
            return mask
        elif isinstance(mask, np.ndarray) and mask.dtype != bool:
            if bits is not None:
                return (np.bitwise_and(mask, bits) == 0)
            else:
                return mask == 0
        else:
            return mask
        
    @staticmethod
    def get_mask_as_int(wave, mask, bits=MASK_ANY):
        """
        Convert any type of mask into an integer mask.
        If the mask is already integer, keep it as is. If it's a boolean
        mask, True means valid values so it has to be inverted before
        converting into an integer where 0 means valid. If it is a slice,
        do similarly as with boolean.
        """

        if mask is None:
            return None
        elif isinstance(mask, slice):
            m = np.full_like(wave, bits, dtype=int)
            m[mask] = 0
            return m
        elif mask.dtype == bool:
            m = np.full_like(wave, bits, dtype=int)
            m[mask] = 0
            return m
        elif mask.dtype == int:
            return mask
        else:
            raise NotImplementedError("Unknown mask format")
        
    @staticmethod
    def get_mask_merged(wave, mask1, mask2, bits1=MASK_DEFAULT, bits2=MASK_DEFAULT):
        mask1 = Spectrum.get_mask_as_int(wave, mask1, bits=bits1)
        mask2 = Spectrum.get_mask_as_int(wave, mask2, bits=bits2)

        if mask1 is None and mask2 is None:
            return None
        elif mask1 is not None and mask2 is None:
            return mask1
        elif mask1 is None and mask2 is not None:
            return mask2
        else:
            return np.bitwise_or(mask1, mask2)

    #endregion
    #region Wave manipulation
        
    @staticmethod
    def get_wave_edges_1d(wave_edges):
        if wave_edges is None:
            return None
        elif wave_edges.ndim == 1:
            return wave_edges
        elif wave_edges.ndim == 2:
            # TODO: this assumes that the bind are adjecent!
            return np.concatenate([wave_edges[0], wave_edges[1][-1:]])
        else:
            raise NotImplementedError()

    @staticmethod
    def get_wave_edges_2d(wave_edges):
        if wave_edges is None:
            return None
        elif wave_edges.ndim == 1:
            return np.stack([wave_edges[:-1], [wave_edges[1:]]], axis=0)
        elif wave_edges.ndim == 2:
            return wave_edges
        else:
            raise NotImplementedError()
        
    #endregion
        
    def set_redshift(self, z):
        """
        Apply Doppler shift by chaning the wave grid only, but not the flux density.
        """

        if self.redshift is not None and self.redshift != 0.0:
            raise ValueError("Initial redshift is not zero, not shifting to z.")

        # Assume zero redshift at start
        self.wave = (1 + z) * self.wave
        if self.wave_edges is not None:
            self.wave_edges = (1 + z) * self.wave_edges
        
        self.redshift = z

        self.append_history(f'Redshift changed from {0.0} to {self.redshift}.')

    def set_restframe(self):
        """
        Correct for Doppler shift by chaning the wave grid only, but not the flux density.
        """

        if self.redshift is None or np.isnan(self.redshift):
            raise ValueError("Unknown redshift, cannot convert to rest-frame.")
        
        self.append_history(f'Redshift changed from {self.redshift} to {0.0}.')

        self.wave = self.wave / (1 + self.redshift)
        if self.wave_edges is not None:
            self.wave_edges = self.wave_edges / (1 + self.redshift)
        self.redshift = 0.0

    #region Resampling

    def apply_resampler_impl(self, resampler):
        # TODO: use MASK_NOWAVE but also look for nan values in wave inside resampler

        self.flux, self.flux_err, mask = resampler.resample_value(self.wave, self.wave_edges, self.flux, self.flux_err)
        self.flux_sky, _, _ = resampler.resample_value(self.wave, self.wave_edges, self.flux_sky)
        self.cont, _, _ = resampler.resample_value(self.wave, self.wave_edges, self.cont)
        self.cont_fit, _, _ = resampler.resample_value(self.wave, self.wave_edges, self.cont_fit)

        # TODO: flux_calibration
        #       flux_model
        
        self.mask = resampler.resample_mask(self.wave, self.wave_edges, self.mask)
        self.weight = resampler.resample_weight(self.wave, self.wave_edges, self.weight)

        return mask

    def apply_resampler(self, resampler, target_wave, target_wave_edges):
        resampler.init(target_wave, target_wave_edges)
        mask = self.apply_resampler_impl(resampler)
        resampler.reset()
        
        self.wave = target_wave
        self.wave_edges = target_wave_edges

        self.merge_mask(mask, Spectrum.MASK_EXTRAPOLATED)

        self.append_history(f'Resampled using a resampler of type `{type(resampler).__name__}`.')

    #endregion

    def trim_wave(self, wlim):
        """
        Trim wavelength to be between limits
        """

        def trim_vector(data, mask):
            if data is None:
                return None
            else:
                return data[mask]

        if self.wave_edges is not None:
            mask = (wlim[0] <= self.wave_edges) & (self.wave_edges <= wlim[1])
        else:
            mask = (wlim[0] <= self.wave) & (self.wave <= wlim[1])

            self.wave = trim_vector(self.wave, mask)
            self.flux = trim_vector(self.flux, mask)
            self.flux_model = trim_vector(self.flux_model, mask)
            self.flux_err = trim_vector(self.flux_err, mask)
            self.flux_sky = trim_vector(self.flux_sky, mask)
            self.flux_calibration = trim_vector(self.flux_calibration, mask)
            self.mask = trim_vector(self.mask, mask)
            self.weight = trim_vector(self.weight, mask)
            self.cont = trim_vector(self.cont, mask)
            self.cont_fit = trim_vector(self.cont_fit, mask)
       
    def zero_mask(self):
        self.flux[self.mask != 0] = 0
        if self.flux_err is not None:
            self.flux_err[self.mask != 0] = 0
        if self.flux_sky is not None:
            self.flux_sky[self.mask != 0] = 0

    def multiply(self, a, silent=True):
        if np.all(np.isfinite(a)):
            self.flux = a * self.flux
            if self.flux_err is not None:
                self.flux_err = a * self.flux_err
            if self.flux_sky is not None:
                self.flux_sky = a * self.flux_sky
            if self.cont is not None:
                self.cont = a * self.cont

            self.append_history(f'Multiplied by {a}.')
        elif not silent:
            raise Exception('Cannot multiply by NaN of Inf')

    def normalize_at(self, lam, value=1.0):
        idx = np.digitize(lam, self.wave)
        flux = self.flux[idx]
        self.multiply(value / flux)

    def normalize_in(self, lam, func=np.median, value=1.0):
        if type(lam[0]) is float:
            # single range
            idx = np.digitize(lam, self.wave)
            fl = self.flux[idx[0]:idx[1]]
        else:
            # list of ranges
            fl = []
            for rlam in lam:
                idx = np.digitize(rlam, self.wave)
                fl += list(self.flux[idx[0]:idx[1]])
            fl = np.array(fl)
        if fl.shape[0] < 2:
            raise Exception('Cannot get wavelength interval')
        fl = func(fl)
        self.multiply(value / fl)

        self.append_history(f'Normalized to unit flux within {lam} using {func.__name__}.')

    def normalize_to_mag(self, filt, mag):
        try:
            m = self.synthmag(filt)
        except Exception as ex:
            print('flux max', np.max(self.flux))
            print('mag', mag)
            raise ex
        DM = mag - m
        D = 10 ** (DM / 5)

        self.multiply(1 / D**2)
        self.mag = mag

    def normalize_by_continuum(self):
        self.cont_model = self.cont.copy()             # Save for later
        self.multiply(1.0 / self.cont)

        self.append_history('Normalized by continuum vector.')

    def apply_noise(self, noise_model, noise_level=None, random_state=None):
        """
        Generate the noise based on a noise model and add to the flux.
        """

        self.flux = noise_model.apply_noise(self.wave, self.flux, self.flux_err, mag=self.mag, noise_level=noise_level, random_state=random_state)

        self.append_history(f'Applied noise model of type `{type(noise_model).__name__}`.')

    def calculate_snr(self, snr):
        # TODO: add more options to specify the mask
        self.snr = snr.get_snr(self.flux, self.flux_err, self.mask_as_bool())

        self.append_history(f'S/N calculated to be {self.snr} using method `{type(snr).__name__}`')

        return self.snr
        
    def redden(self, extval=None):
        extval = extval or self.ext
        self.ext = extval
        
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux, keepneg=True)
        # Cardelli, Clayton, & Mathis (1989, ApJ, 345, 245) R_V = 3.10.
        obs = spec * pysynphot.reddening.Extinction(extval, 'mwavg')
        self.flux = obs.flux

        self.append_history(f'Applied reddening of {extval} using `mwavg`.')

    def deredden(self, extval=None):
        extval = extval or self.ext
        self.redden(-extval)

    def synthflux(self, filter):
        # TODO: Calculate error from error array?

        filt = pysynphot.spectrum.ArraySpectralElement(filter.wave, filter.thru, waveunits='angstrom')
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux, keepneg=True, fluxunits='flam')
        
        # Binning of the filter should be the same as the spectrum but trimmed
        # to the coverage of the spectrum.
        # Setting binset manually will supress the warning from pysynphot
        mask = (filt.wave.min() <= spec.wave) & (spec.wave <= filt.wave.max())
        filt.binset = spec.wave[mask]

        try:
            obs = pysynphot.observation.Observation(spec, filt)
            return obs.effstim('Jy')
        except Exception as ex:
            logging.debug(f'Synthetic flux in filter `{filter.name}` turned out to be negative.')
            return np.nan

    def synthmag(self, filter, norm=1.0):
        flux = norm * self.synthflux(filter)
        return -2.5 * np.log10(flux) + 8.90

    def get_wlim_slice(self, wlim, buffer=None):
        if wlim is not None:
            if buffer is not None:
                wlim = (wlim[0] - buffer, wlim[1] + buffer)

            if self.mask is not None:
                # NOTE: Here we assume that wave masks can only appear at the beginning
                #       and end of the grid, otherwise this won't work properly

                mask = self.mask_as_bool(bits=Spectrum.MASK_NOWAVE)
                idx = np.digitize(wlim, self.wave[mask])
                
                # Slice must refer to original wave grid
                return slice(*np.arange(self.wave.size + 1, dtype=int)[idx])
            else:
                return slice(*np.digitize(wlim, self.wave))
        else:
            return slice(None)

    def convolve_gaussian(self, dlambda=None, vdisp=None, wlim=None):
        """
        Convolve with (a potentially wavelength dependent) Gaussian kernel)
        :param dlam: dispersion in units of Angstrom
        :param wave: wavelength in Ansgstroms
        :param wlim: limit convolution between wavelengths
        :return:
        """

        if dlambda is not None:
            psf = GaussPsf(np.array([self.wave[0], self.wave[-1]]), None, np.array([dlambda, dlambda]))
        elif vdisp is not None:
            psf = VelocityDispersion(vdisp)
        else:
            raise Exception('dlambda or vdisp must be specified')

        idx = self.get_wlim_slice(wlim)
        s = psf.get_optimal_size(self.wave[idx])
        self.convolve_psf(psf, size=s, wlim=wlim)

    def convolve_psf(self, psf, size=None, wlim=None):
        def set_array(a, b):
            r = a.copy()
            r[idx][s] = b
            return r

        # Determine kernel width and buffer convolution wavelength range
        width = psf.get_width(self.wave)
        idx = self.get_wlim_slice(wlim, buffer=width)
        wave = self.wave[idx]

        flux = [self.flux[idx]]
        if self.cont is not None:
            flux.append(self.cont[idx])

        error = []
        if self.flux_err is not None:
            error.append(self.flux_err[idx])

        if isinstance(psf, Psf):
            w, flux, error, shift = psf.convolve(wave, values=flux, errors=error, size=size, normalize=True)
            s = np.s_[shift:wave.shape[0] - shift]
        else:
            raise NotImplementedError()

        self.flux = set_array(self.flux, flux.pop(0))
        if self.cont is not None:
            self.cont = set_array(self.cont, flux.pop(0))
        
        if self.flux_err is not None:
            self.flux_err = set_array(self.flux_err, error.pop(0))

        # TODO: try to estimate the resolution or give name to PSF to be able to distinguish
        self.append_history(f'Convolved with PSF of type `{type(psf).__name__}` with width of {2 * shift + 1}.')
           
    # TODO: Move to spectrum tools
    @staticmethod
    def running_filter(wave, data, func, dlambda=None, vdisp=None):
        # TODO: use get_dispersion
        # Don't care much about edges here, they'll be trimmed when rebinning
        if dlambda is not None and vdisp is not None:
            raise Exception('Only one of dlambda and vdisp can be specified')
        elif dlambda is None and vdisp is None:
            vdisp = Constants.DEFAULT_FILTER_VDISP

        if vdisp is not None:
            z = vdisp * 1e3 / Physics.c

        ndata = np.empty(data.shape)
        for i in range(len(wave)):
            if isinstance(dlambda, collections.abc.Iterable):
                mask = (wave[i] - dlambda[0] <= wave) & (wave < wave[i] + dlambda[1])
            elif dlambda is not None:
                mask = (wave[i] - dlambda <= wave) & (wave < wave[i] + dlambda)
            else:
                mask = ((1 - z) * wave[i] < wave) & (wave < (1 + z) * wave[i])

            if len(data.shape) == 1:
                if mask.size < 2:
                    ndata[i] = data[i]
                else:
                    ndata[i] = func(data[mask])
            elif len(data.shape) == 2:
                if mask.size < 2:
                    ndata[:, i] = data[:, i]
                else:
                    ndata[:, i] = func(data[:, mask], axis=1)
            else:
                raise NotImplementedError()

        return ndata

    def high_pass_filter(self, func=np.median, dlambda=None, vdisp=None):
        # TODO: error array?
        self.flux -= Spectrum.running_filter(self.wave, self.flux, func, dlambda=dlambda, vdisp=vdisp)
        if self.flux_sky is not None:
            self.flux_sky -= Spectrum.running_filter(self.wave, self.flux_sky, func, vdisp)

    # TODO: Move to stellarmod mixin
    def fit_envelope_chebyshev(self, wlim=None, iter=10, order=4, clip_top=3.0, clip_bottom=0.1):
        # TODO: use mask from spectrum
        wave = self.wave
        flux = np.copy(self.flux)
        if self.flux_err is not None:
            # TODO: maybe do some smoothing or filter out 0
            weight = 1 / self.flux_err
        else:
            weight = np.full(self.flux.shape, 1)

        mask = np.full(flux.shape, False)
        if wlim is not None and wlim[0] is not None and wlim[1] is not None:
            widx = np.digitize(wlim, wave)
            mask[:widx[0]] = True
            mask[widx[1]:] = True

        # Early stop if we have lost too many bins to do a meaningful fit
        while iter > 0 and np.sum(~mask) > 100:
            p = np.polynomial.chebyshev.chebfit(wave[~mask], flux[~mask], order, w=weight[~mask])
            c = np.polynomial.chebyshev.chebval(wave, p)
            s = np.sqrt(np.var(flux[~mask] - c[~mask]))
            mask |= ((flux > c + clip_top * s) | (flux < c - clip_bottom * s))

            iter -= 1

        return p, c

    def apply_calibration(self, calibration):
        calibration.apply_calibration(self)

        self.append_history(f'Applied flux calibration of type `{type(calibration).__name__}`.')

    def load(self, filename):
        raise NotImplementedError()

    def plot(self, ax=None, xlim=Constants.DEFAULT_PLOT_WAVE_RANGE, ylim=None, labels=True):
        ax = self.plot_getax(ax, xlim, ylim)
        ax.plot(self.wave, self.flux)
        if labels:
            ax.set_xlabel(r'$\lambda$ [A]')
            ax.set_ylabel(r'$F_\lambda$ [erg s-1 cm-2 A-1]')

        return ax

    def print_info(self):
        for p in self.get_param_names():
            print(p, getattr(self, p))

    def wave_in_unit(self, unit):
        if self.wave is None:
            return None
        elif unit == 'AA':
            return self.wave, self.wave_edges
        elif unit == 'nm':
            return Physics.angstrom_to_nm(self.wave), Physics.angstrom_to_nm(self.wave_edges)
        else:
            raise NotImplementedError()
        
    def __flux_in_unit(self, flux, unit):
        def conv(factor, vector):
            return factor * vector if vector is not None else None

        if flux is None:
            return None
        elif unit == 'erg s-1 cm-2 A-1':
            return flux
        elif unit in ['nJy', 'Jy', 'erg s-1 cm-2 Hz-1']:
            flux = Physics.flam_to_fnu(self.wave, flux)
            if unit == 'nJy':
                return conv(1e32, flux)
            elif unit == 'Jy':
                return conv(1e23, flux)
            elif unit == 'erg s-1 cm-2 Hz-1':
                return flux
        else:
            return NotImplementedError()
        
    def flux_in_unit(self, unit):
        return self.__flux_in_unit(self.flux, unit), self.__flux_in_unit(self.flux_err, unit)
        
    def cont_in_unit(self, unit):
        return self.__flux_in_unit(self.cont, unit)

    def get_id_string(self):
        return f'{self.id}'
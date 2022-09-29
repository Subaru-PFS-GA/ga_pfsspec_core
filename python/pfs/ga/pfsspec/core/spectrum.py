import logging
import numpy as np
import numbers
import collections
import matplotlib.pyplot as plt
import pysynphot
import pysynphot.binning
import pysynphot.spectrum
import pysynphot.reddening

from pfs.ga.pfsspec.core.util.copy import *
from pfs.ga.pfsspec.core.pfsobject import PfsObject
from pfs.ga.pfsspec.core.obsmod.psf import *
from .physics import Physics
from .constants import Constants

class Spectrum(PfsObject):
    def __init__(self, orig=None):
        super(Spectrum, self).__init__(orig=orig)
        
        if not isinstance(orig, Spectrum):
            self.index = None
            self.id = 0
            self.redshift = 0.0
            self.redshift_err = 0.0

            # TODO: these should go elsewhere, they apply to observed spectra only
            self.exp_count = None
            self.exp_time = None
            self.seeing = None
            self.extinction = None
            self.target_zenith_angle = None
            self.target_field_angle = None
            self.moon_zenith_angle = None
            self.moon_target_angle = None
            self.moon_phase = None
            self.snr = None
            self.mag = None
            ###

            self.wave = None
            self.wave_edges = None
            self.flux = None                    # Flux (+noise when frozen)
            self.flux_model = None              # Flux (model, when noise is frozen)
            self.flux_err = None                # Flux error (sigma, not squared)
            self.flux_sky = None                # Background flux (sky + moon, no detector)
            self.flux_calibration = None
            self.mask = None
            self.cont = None
            self.cont_fit = None
            self.random_seed = None

            self.resolution = None
            self.is_wave_regular = None
            self.is_wave_lin = None
            self.is_wave_log = None
        else:
            self.index = orig.index
            self.id = orig.id

            # TODO: move these from the base class
            self.redshift = orig.redshift
            self.redshift_err = orig.redshift_err
            self.exp_count = orig.exp_count
            self.exp_time = orig.exp_time
            self.seeing = orig.seeing
            self.extinction = orig.extinction
            self.target_zenith_angle = orig.target_zenith_angle
            self.target_field_angle = orig.target_field_angle
            self.moon_zenith_angle = orig.moon_zenith_angle
            self.moon_target_angle = orig.moon_target_angle
            self.moon_phase = orig.moon_phase
            self.snr = orig.snr
            self.mag = orig.mag
            ###
            
            self.wave = safe_deep_copy(orig.wave)
            self.wave_edges = safe_deep_copy(orig.wave_edges)
            self.flux = safe_deep_copy(orig.flux)
            self.flux_model = safe_deep_copy(orig.flux_model)
            self.flux_err = safe_deep_copy(orig.flux_err)
            self.flux_sky = safe_deep_copy(orig.flux_sky)
            self.flux_calibration = safe_deep_copy(orig.flux_calibration)
            self.mask = safe_deep_copy(orig.mask)
            self.cont = safe_deep_copy(orig.cont)
            self.cont_fit = orig.cont_fit
            self.random_seed = orig.random_seed

            self.resolution = orig.resolution
            self.is_wave_regular = orig.is_wave_regular
            self.is_wave_lin = orig.is_wave_lin
            self.is_wave_log = orig.is_wave_log

    def get_param_names(self):
        return ['id',
                'redshift',
                'redshift_err',
                'exp_count',
                'exp_time',
                'extinction',
                'target_zenith_angle',
                'target_field_angle',
                'moon_zenith_angle',
                'moon_target_angle',
                'moon_phase',
                'snr',
                'mag',
                'cont_fit',
                'random_seed']

    def get_params(self):
        params = {}
        for p in self.get_param_names():
            params[p] = getattr(self, p)
        return params

    def set_params(self, params):
        for p in self.get_param_names():
            if p in params:
                setattr(self, p, params[p])

    def get_params_as_datarow(self):
        row = {}
        for p in self.get_param_names():
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

    def set_restframe(self):
        """
        Correct for Doppler shift by chaning the wave grid only, but not the flux density.
        """

        if self.redshift is None or np.isnan(self.redshift):
            raise ValueError("Unknown redshift, cannot convert to rest-frame.")

        self.wave = self.wave / (1 + self.redshift)
        if self.wave_edges is not None:
            self.wave_edges = self.wave_edges / (1 + self.redshift)
        self.redshift = 0.0

    def apply_resampler(self, resampler, wave, wave_edges):
        resampler.init(wave, wave_edges)
        self.apply_resampler_impl(resampler)
        resampler.reset()

        self.wave = wave
        self.wave_edges = wave_edges
    
    def apply_resampler_impl(self, resampler):

        self.flux, self.flux_err = resampler.resample_value(self.wave, self.wave_edges, self.flux, self.flux_err)
        self.flux_sky, _ = resampler.resample_value(self.wave, self.wave_edges, self.flux_sky)
        self.cont, _ = resampler.resample_value(self.wave, self.wave_edges, self.cont)
        self.cont_fit, _ = resampler.resample_value(self.wave, self.wave_edges, self.cont_fit)
        
        self.mask = resampler.resample_mask(self.wave, self.wave_edges, self.mask)
       
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
        self.multiply(1.0 / self.cont)

    def apply_noise(self, noise_model, noise_level=None, random_seed=None):
        """
        Generate the noise based on a noise model and add to the flux.
        """

        self.flux = noise_model.apply_noise(self.wave, self.flux, self.flux_err, mag=self.mag, noise_level=noise_level, random_seed=random_seed)

    def calculate_snr(self, snr):
        self.snr = snr.get_snr(self.flux, self.flux_err)
        
    def redden(self, extval=None):
        extval = extval or self.extinction
        self.extinction = extval
        
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=self.wave, flux=self.flux, keepneg=True)
        # Cardelli, Clayton, & Mathis (1989, ApJ, 345, 245) R_V = 3.10.
        obs = spec * pysynphot.reddening.Extinction(extval, 'mwavg')
        self.flux = obs.flux

    def deredden(self, extval=None):
        extval = extval or self.extinction
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
            flux = obs.effstim('Jy')
        except Exception as ex:
            raise ex
        return flux

    def synthmag(self, filter, norm=1.0):
        flux = norm * self.synthflux(filter)
        return -2.5 * np.log10(flux) + 8.90

    def get_wlim_slice(self, wlim):
        if wlim is not None:
            return slice(*np.digitize(wlim, self.wave))
        else:
            return slice(None)

    def convolve_gaussian(self,  dlambda=None, vdisp=None, wlim=None):
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

        idx = self.get_wlim_slice(wlim)

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
            if isinstance(dlambda, collections.Iterable):
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
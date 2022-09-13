import logging
import numpy as np
import numbers
from scipy.interpolate import interp1d
import collections
import matplotlib.pyplot as plt
import pysynphot
import pysynphot.binning
import pysynphot.spectrum
import pysynphot.reddening

from pfs.ga.pfsspec.core.pfsobject import PfsObject
from .physics import Physics
from .constants import Constants
# from pfs.ga.pfsspec.obsmod.psf import Psf
# from pfs.ga.pfsspec.obsmod.pcapsf import PcaPsf
# from pfs.ga.pfsspec.obsmod.gausspsf import GaussPsf

class Spectrum(PfsObject):
    def __init__(self, orig=None):
        super(Spectrum, self).__init__(orig=orig)
        
        if not isinstance(orig, Spectrum):
            self.index = None
            self.id = 0
            self.redshift = 0.0
            self.redshift_err = 0.0

            # TODO: these should go elsewhere
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
            self.flux = None
            self.flux_err = None
            self.flux_sky = None
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
            
            self.wave = np.copy(orig.wave)
            self.wave_edges = np.copy(orig.wave_edges)
            self.flux = np.copy(orig.flux)
            self.flux_err = np.copy(orig.flux_err)
            self.flux_sky = np.copy(orig.flux_sky)
            self.mask = np.copy(orig.mask)
            self.cont = np.copy(orig.cont)
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

    # TODO: Move to spectrum tools
    @staticmethod
    def generate_noise(flux, noise, error=None, random_seed=None):
        # Noise have to be reproducible when multiple datasets with different
        # post-processing are generated for autoencoders
        if random_seed is not None:
            np.random.seed(random_seed)

        if error is not None:
            # If error vector is present, use as sigma for additive noise
            err = noise * np.random.normal(size=flux.shape) * error
            return flux + err
        else:
            # Simple multiplicative noise, one random number per bin
            err = np.random.normal(1, noise, flux.shape)
            return flux * err

    def calculate_snr(self, weight=1.0):
        if self.flux_err is not None:
            # Make sure flux value is reasonable, otherwise exclude from averate SNR
            f = (self.flux_err <= 10 * self.flux) & (self.flux_err > 0)
            self.snr = np.mean(np.abs(self.flux[f] / self.flux_err[f])) / weight
        else:
            self.snr = 0
        
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

    # TODO: Move to obsmod mixin
    @staticmethod
    def generate_calib_bias(wave, bandwidth=200, amplitude=0.05):
        """
        Simulate claibration error but multiplying with a slowly changing function

        :param bandwidth:
        :param amplitude:
        """

        def white_noise(N):
            s = 2 * np.exp(1j * np.random.rand(N // 2 + 1) * 2 * np.pi)
            noise = np.fft.irfft(s)
            return noise[:N]

        def wiener(N):
            step = white_noise(N + 1)
            walk = np.cumsum(step)[:N]
            return walk

        def damped_walk(N, alpha=1, D=1):
            step = white_noise(N)
            walk = np.zeros(N)
            for i in range(1, N):
                walk[i] = walk[i - 1] - alpha * walk[i - 1] + D * step[i]
            return walk

        def gauss(x, mu, sigma):
            return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

        def lowpass_filter(x, y, sigma):
            mid = np.mean(x)
            idx = np.digitize([mid - 3 * sigma, mid + 3 * sigma], wave)
            k = gauss(x[idx[0]:idx[1]], mid, sigma)
            k /= np.sum(k)
            return np.convolve(y, k, mode='same')

        bias = wiener(wave.shape[0])
        bias = lowpass_filter(wave, bias, 200)
        min = np.min(bias)
        max = np.max(bias)
        bias = 1.0 - amplitude * (bias - min) / (max - min)

        return bias

    # TODO: Move to obsmod mixin
    def add_calib_bias(self, bandwidth=200, amplitude=0.05):
        bias = Spectrum.generate_calib_bias(self.wave, bandwidth=bandwidth, amplitude=amplitude)
        self.multiply(bias)

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
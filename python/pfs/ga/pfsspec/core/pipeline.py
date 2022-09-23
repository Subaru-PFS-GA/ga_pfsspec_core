import numpy as np
import numbers
from scipy.interpolate import interp1d
import collections
import pysynphot
import pysynphot.binning
import pysynphot.spectrum
import pysynphot.reddening

from .physics import Physics
from .constants import Constants
from .pfsobject import PfsObject
from .filter import Filter
from .spectrum import Spectrum
from .psf.psf import Psf

class Pipeline(PfsObject):
    """
    Implements a basic spectrum processing pipeline with minimum functionality
    to redshift, convolve, resample and normalize spectra. These are intended to
    perform basic operations during model spectrum import and not to simulate
    observations.
    """

    def __init__(self, orig=None):
        if not isinstance(orig, Pipeline):
            self.random_state = None

            self.restframe = False          # When true, use value from spectra to convert to restframe
            self.redshift = None            # When true, use value from spectra to shift to certain redshift,
                                            # when value, use value to redshift spectrum to
            self.conv = None                # Convolution method
            self.conv_sigma = None          # Gaussian convolution sigma in wavelength units
            self.conv_resolution = None     # Convolution to a given resolution
            self.wave = None                # When set, resample to this grid
            self.wave_edges = None
            self.wave_lin = False
            self.wave_log = False
            self.wave_bins = None
            self.norm = None                # Post process normalization method
            self.norm_wave = None           # Post process normalization wavelength range
        else:
            self.random_state = orig.random_state

            self.conv = orig.conv
            self.conv_sigma = orig.conv_sigma
            self.conv_resolution = orig.conv_resolution
            self.wave = orig.wave
            self.wave_edges = orig.wave_edges
            self.wave_lin = orig.wave_lin
            self.wave_log = orig.wave_log
            self.wave_bins = orig.wave_bins
            self.norm = orig.norm
            self.norm_wave = orig.norm_wave

    def add_args(self, parser):
        parser.add_argument('--restframe', action='store_true', help='Convert to rest-frame.\n')
        parser.add_argument('--redshift', type=float, nargs='?', help='Shift to given redshift.\n')
        parser.add_argument('--conv', type=str, choices=[Constants.CONVOLUTION_RESOLUTION, Constants.CONVOLUTION_GAUSS], help='Convolution method.\n')
        parser.add_argument('--conv-sigma', type=float, help='Gaussian convolution sigma in AA.\n')
        parser.add_argument('--conv-resolution', type=float, help='Gaussian convolution to resolution.\n')

        parser.add_argument('--wave', type=float, nargs=2, default=None, help='Wavelength range.\n')
        parser.add_argument('--wave-bins', type=int, help='Number of wavelength bins.\n')
        parser.add_argument('--wave-lin', action='store_true', help='Linear wavelength binning.\n')
        parser.add_argument('--wave-log', action='store_true', help='Logarithmic wavelength binning.\n')

        parser.add_argument('--norm', type=str, default=None, help='Normalization method\n')
        parser.add_argument('--norm-wave', type=float, nargs=2, default=[4200, 6500], help='Normalization method\n')

    def init_from_args(self, args):
        self.restframe = self.get_arg('restframe', self.restframe, args)
        if self.is_arg('redshift'):
            raise NotImplementedError()

        self.conv = self.get_arg('conv', self.conv, args)
        self.conv_sigma = self.get_arg('conv_sigma', self.conv_sigma, args)
        self.conv_resolution = self.get_arg('conv_resolution', self.conv_resolution, args)

        # If only wave limits are specified, the input grid will be truncated
        # If the number of bins are specified as well, resampling will be done
        self.wave = self.get_arg('wave', self.wave, args)
        self.wave_bins = self.get_arg('wave_bins', self.wave_bins, args)
        self.wave_lin = self.get_arg('wave_lin', self.wave_lin, args)
        self.wave_log = self.get_arg('wave_log', self.wave_log, args)
        
        self.norm = self.get_arg('norm', self.norm, args)
        self.norm_wave = self.get_arg('norm_wave', self.norm_wave, args)
      
    def is_constant_wave(self):
        # Returns true when the wavelength grid is the same for all processed spectra
        return not self.restframe and self.redshift is None

    def get_wave(self, spec: Spectrum = None, **kwargs):
        if isinstance(self.wave, np.ndarray):
            return self.wave, self.wave_edges
        elif spec is None:
            return self.get_wave_const()
        else:
            return self.get_wave_spec(spec, **kwargs)

    def get_wave_const(self):
        if self.wave_lin and self.wave_log:
            raise ValueError("Only one of --wave-lin and --wave-log must be specified.")
        
        if (self.wave_lin or self.wave_log) and (self.wave is None or self.wave_bins is None):
            raise ValueError("--wave and --wave-bins must be specified when --wave-lin or --wave-log is set.")
        
        if self.wave_bins is not None and not self.wave_lin and not self.wave_log:
            raise ValueError("--wave-lin or --wave-log must be specified when --wave-bin is set.")

        if self.wave_lin:
            # Linear binning with a constant grid
            edges = np.linspace(self.wave[0], self.wave[1], self.wave_bins + 1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            self.wave = centers
            self.wave_edges = np.stack([edges[:-1], edges[1:]])
            return self.wave, self.wave_edges
        elif self.wave_log:
            # Logarithmic binning with a constant grid
            edges = np.linspace(np.log10(self.wave[0]), np.log10(self.wave[1]), self.wave_bins + 1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            self.wave = 10 ** centers
            self.wave_edges = 10 ** np.stack([edges[:-1], edges[1:]])
            return self.wave, self.wave_edges
        else:
            raise ValueError("No wavelength limits are specified.")
    
    def get_wave_spec(self, spec, **kwargs):
        # Take the grid from the spectrum but limit it to the range
        if spec.wave is not None and spec.wave_edges is not None:
            pass
        elif spec.wave is not None:
            pass
        else:
            raise ValueError("Spectrum does not specify a wave grid.")

        # TODO: implement logic to restframe/redshift grid and then truncate to limit
        raise NotImplementedError()

    def get_wave_count(self, spec=None, **kwargs):
        if spec is None:
            centers, _ = self.get_wave_const()
            return centers.shape[0]
        else:
            centers, _ = self.get_wave_spec(spec, **kwargs)
            return centers.shape[0]

    def run(self, spec, **kwargs):
        self.run_step_restframe(spec, **kwargs)
        self.run_step_redshift(spec, **kwargs)
        self.run_step_convolution(spec, **kwargs)
        self.run_step_rebin(spec, **kwargs)
        self.run_step_normalize(spec, **kwargs)

    def run_step_restframe(self, spec: Spectrum, **kwargs):
        # If the spectrum is already redshifted, convert it to restframe
        z = spec.redshift
        if z is not None and not np.isnan(z):
            spec.set_restframe()

    def run_step_redshift(self, spec: Spectrum, **kwargs):
        z = self.get_arg('redshift', self.redshift, kwargs)
        if z is not None and not np.isnan(z) and z != 0.0:
            spec.set_redshift(z)

    def run_step_rebin(self, spec, **kwargs):
        if self.wave is not None:
            wave, wave_edges = self.get_wave()
            self.rebin(spec, wave, wave_edges)

    def rebin(self, spec, wave, wave_edges):
        spec.rebin(wave, wave_edges)

    #region Convolution

    # @staticmethod
    # def convolve_vector(wave, data, kernel_func, dlambda=None, vdisp=None, wlim=None):
    #     """
    #     Convolve a data vector (flux) with a wavelength-dependent kernel provided as a callable.
    #     This is very generic, assumes no regular binning and constant-width kernel but bins should
    #     be similarly sized.
    #     :param data:
    #     :param kernel:
    #     :param dlam:
    #     :param wlim:
    #     :return:
    #     """

    #     # TODO: if bins are not similarly sized we need to compute bin widths and take that into
    #     # account

    #     def get_dispersion(dlambda=None, vdisp=None):
    #         if dlambda is not None and vdisp is not None:
    #             raise Exception('Only one of dlambda and vdisp can be specified')
    #         if dlambda is not None:
    #             if isinstance(dlambda, collections.Iterable):
    #                 return dlambda, None
    #             else:
    #                 return [dlambda, dlambda], None
    #         elif vdisp is not None:
    #             z = Physics.vel_to_z(vdisp)
    #             return None, z
    #         else:
    #             z = Physics.vel_to_z(Constants.DEFAULT_FILTER_VDISP)
    #             return None, z

    #     # Start and end of convolution
    #     if wlim is not None:
    #         idx_lim = np.digitize(wlim, wave)
    #     else:
    #         idx_lim = [0, wave.shape[0] - 1]

    #     # Dispersion determines the width of the kernel to be taken into account
    #     dlambda, z = get_dispersion(dlambda, vdisp)

    #     # Construct results and fill-in parts that are not part of the original
    #     if not isinstance(data, tuple) and not isinstance(data, list):
    #         data = [data, ]
    #     res = [ np.zeros(d.shape) for d in data ]
    #     for d, r in zip(data, res):
    #         r[:idx_lim[0]] = d[:idx_lim[0]]
    #         r[idx_lim[1]:] = d[idx_lim[1]:]

    #     # Compute convolution
    #     for i in range(idx_lim[0], idx_lim[1]):
    #         if dlambda is not None:
    #             idx = np.digitize([wave[i] - dlambda[0], wave[i] + dlambda[0]], wave)
    #         else:
    #             idx = np.digitize([(1 - z) * wave[i], (1 + z) * wave[i]], wave)

    #         # Evaluate kernel
    #         k = kernel_func(wave[i], wave[idx[0]:idx[1]])

    #         # Sum up
    #         for d, r in zip(data, res):
    #             r[i] += np.sum(k * d[idx[0]:idx[1]])

    #     return res

    # @staticmethod
    # def convolve_vector_varying_kernel(wave, data, kernel_func, size=None, wlim=None):
    #     """
    #     Convolve with a kernel that varies with wavelength. Kernel is computed
    #     by a function passed as a parameter.
    #     """

    #     # Get a test kernel from the middle of the wavelength range to have its size
    #     kernel = kernel_func(wave[wave.shape[0] // 2], size=size)

    #     # Start and end of convolution
    #     # Outside this range, original values will be used
    #     if wlim is not None:
    #         idx = np.digitize(wlim, wave)
    #     else:
    #         idx = [0, wave.shape[0]]

    #     idx_lim = [
    #         max(idx[0], kernel.shape[0] // 2),
    #         min(idx[1], wave.shape[0] - kernel.shape[0] // 2)
    #     ]

    #     # Construct results
    #     if not isinstance(data, tuple) and not isinstance(data, list):
    #         data = [data, ]
    #     res = [np.zeros(d.shape) for d in data]

    #     # Do the convolution
    #     # TODO: can we optimize it further?
    #     offset = 0 - kernel.shape[0] // 2
    #     for i in range(idx_lim[0], idx_lim[1]):
    #         kernel = kernel_func(wave[i], size=size)
    #         s = slice(i + offset, i + offset + kernel.shape[0])
    #         for d, r in zip(data, res):
    #             z = d[i] * kernel
    #             r[s] += z

    #     # Fill-in parts that are not convolved
    #     for d, r in zip(data, res):
    #         r[:idx_lim[0] - offset] = d[:idx_lim[0] - offset]
    #         r[idx_lim[1] + offset:] = d[idx_lim[1] + offset:]

    #     return res

    def convolve_gaussian_log(self, spec, lambda_ref=5000,  dlambda=None, vdisp=None, wlim=None):
        """
        Convolve with a Gaussian filter, assume logarithmic binning in wavelength
        so the same kernel can be used across the whole spectrum
        :param dlambda:
        :param vdisp:
        :param wlim:
        :return:
        """

        def convolve_vector(wave, data, kernel, wlim=None):
            # Start and end of convolution
            if wlim is not None:
                idx = np.digitize(wlim, wave)
                idx_lim = [max(idx[0] - kernel.shape[0] // 2, 0), min(idx[1] + kernel.shape[0] // 2, wave.shape[0] - 1)]
            else:
                idx_lim = [0, wave.shape[0] - 1]

            # Construct results
            if not isinstance(data, tuple) and not isinstance(data, list):
                data = [data, ]
            res = [np.empty(d.shape) for d in data]

            for d, r in zip(data, res):
                r[idx_lim[0]:idx_lim[1]] = np.convolve(d[idx_lim[0]:idx_lim[1]], kernel, mode='same')

            # Fill-in parts that are not convolved to save a bit of computation
            if wlim is not None:
                for d, r in zip(data, res):
                    r[:idx[0]] = d[:idx[0]]
                r[idx[1]:] = d[idx[1]:]

            return res

        data = [spec.flux, ]
        if spec.cont is not None:
            data.append(spec.cont)

        if dlambda is not None:
            pass
        elif vdisp is not None:
            dlambda = lambda_ref * vdisp / Physics.c * 1000 # km/s
        else:
            raise Exception('dlambda or vdisp must be specified')

        # Make sure kernel size is always an odd number, starting from 3
        idx = np.digitize(lambda_ref, spec.wave)
        i = 1
        while lambda_ref - 4 * dlambda < spec.wave[idx - i] or \
              spec.wave[idx + i] < lambda_ref + 4 * dlambda:
            i += 1
        idx = [idx - i, idx + i]

        wave = spec.wave[idx[0]:idx[1]]
        k = 0.3989422804014327 / dlambda * np.exp(-(wave - lambda_ref) ** 2 / (2 * dlambda ** 2))
        k = k / np.sum(k)

        res = convolve_vector(spec.wave, data, k, wlim=wlim)

        spec.flux = res[0]
        if spec.cont is not None:
            spec.cont = res[1]

    def convolve_varying_kernel(self, spec, kernel_func, size=None, wlim=None):
        # NOTE: this function does not take model resolution into account!

        def convolve_vectors(wave, data, kernel_func, size=None, wlim=None):
            """
            Convolve with a kernel that varies with wavelength. Kernel is computed
            by a function passed as a parameter.
            """

            # Get a test kernel from the middle of the wavelength range to have its size
            kernel = kernel_func(wave[wave.shape[0] // 2], size=size)

            # Start and end of convolution
            # Outside this range, original values will be used
            if wlim is not None:
                idx = np.digitize(wlim, wave)
            else:
                idx = [0, wave.shape[0]]

            idx_lim = [
                max(idx[0], kernel.shape[0] // 2),
                min(idx[1], wave.shape[0] - kernel.shape[0] // 2)
            ]

            # Construct results
            if not isinstance(data, tuple) and not isinstance(data, list):
                data = [data, ]
            res = [np.zeros(d.shape) for d in data]

            # Do the convolution
            # TODO: can we optimize it further?
            offset = 0 - kernel.shape[0] // 2
            for i in range(idx_lim[0], idx_lim[1]):
                kernel = kernel_func(wave[i], size=size)
                s = slice(i + offset, i + offset + kernel.shape[0])
                for d, r in zip(data, res):
                    z = d[i] * kernel
                    r[s] += z

            # Fill-in parts that are not convolved
            for d, r in zip(data, res):
                r[:idx_lim[0] - offset] = d[:idx_lim[0] - offset]
                r[idx_lim[1] + offset:] = d[idx_lim[1] + offset:]

            return res

        data = [spec.flux, ]
        if spec.cont is not None:
            data.append(spec.cont)

        res = convolve_vectors(spec.wave, data, kernel_func, size=size, wlim=wlim)

        spec.flux = res[0]
        if spec.cont is not None:
            spec.cont = res[1]

    # TODO: Is it used with both, a Psf object and a number? Separate!
    def convolve_psf(self, spec, psf, model_res, wlim):

        # NOTE: This a more generic convolution step than what's done
        #       inside the observation model which only accounts for
        #       the instrumental effects.

        # TODO: what to do with model resolution?
        # TODO: add option to do convolution pre/post resampling
        raise NotImplementedError()

        # Note, that this is in addition to model spectrum resolution
        # if isinstance(psf, Psf):
        #     spec.convolve_psf(psf, wlim)
        # elif isinstance(psf, numbers.Number) and model_res is not None:
        #     sigma = psf
        #     spec.convolve_gaussian()

        #     if model_res is not None:
        #         # Resolution at the middle of the detector
        #         fwhm = 0.5 * (wlim[0] + wlim[-1]) / model_res
        #         ss = fwhm / 2 / np.sqrt(2 * np.log(2))
        #         sigma = np.sqrt(sigma ** 2 - ss ** 2)
        #     self.convolve_gaussian(spec, dlambda=sigma, wlim=wlim)
        # elif isinstance(psf, numbers.Number):
        #     # This is a simplified version when the model resolution is not known
        #     # TODO: review this
        #     self.convolve_gaussian_log(spec, 5000, dlambda=psf)
        # else:
        #     raise NotImplementedError()

    def run_step_convolution(self, spec, **kwargs):
        if self.conv == Constants.CONVOLUTION_RESOLUTION:
            # Degrade resolution with Gaussian kernel

            if spec.resolution is None or self.conv_resolution is None:
                raise Exception('Cannot determine resolution for convolution')

            if spec.wave_edges is None:
                raise Exception('Wavelength bin edges are not available.')

            # Determine down-convolution kernel width in the middle of the
            # wavelength coverage of the spectrum
            wave_ref_idx = spec.wave.shape[0] // 2
            wave_ref = spec.wave[wave_ref_idx]
            sigma_orig = wave_ref / spec.resolution
            sigma_new = wave_ref / self.conv_resolution
            sigma_kernel = np.sqrt(sigma_new ** 2 - sigma_orig ** 2)
            
            if spec.is_wave_regular is not None and spec.is_wave_regular:
                if spec.is_wave_log is not None and spec.is_wave_log:
                    self.convolve_gaussian_log(spec, lambda_ref=wave_ref, dlambda=sigma_kernel)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

        elif self.conv is not None:
            raise NotImplementedError()

    #endregion

    def run_step_normalize(self, spec, **kwargs):
        # Normalization in wavelength range overrides previous normalization
        wave = self.norm_wave or (spec.wave[0], spec.wave[-1])
        if self.norm == 'none':
            pass
        elif self.norm == 'median':
            spec.normalize_in([wave], np.median, value=0.5)
        elif self.norm == 'mean':
            spec.normalize_in([wave], np.mean, value=0.5)
        elif self.norm == 'integral':
            spec.normalize_in([wave], np.sum, value=1.0)
        elif self.norm is not None:
            raise NotImplementedError()

    def set_if_valid_param(self, spec, name, **kwargs):
        if name in kwargs and kwargs[name] is not None and not np.isnan(kwargs[name]):
            setattr(spec, name, kwargs[name])
import numpy as np
from scipy.interpolate import interp1d

from .sampling import Parameter
from .physics import Physics
from .pfsobject import PfsObject
from .spectrum import Spectrum
from .obsmod.psf import GaussPsf, PcaPsf
from .obsmod.resampling import Interp1dResampler, FluxConservingResampler

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
            self.conv_gauss = None

            self.psf = None                 # PSF to be used for convolution
            
            self.wave = None                # When set, resample to this grid
            self.wave_edges = None
            self.wave_lin = False
            self.wave_log = False
            self.wave_bins = None
            self.wave_resampler = FluxConservingResampler()

            self.norm = None                # Post process normalization method
            self.norm_wave = None           # Post process normalization wavelength range
        else:
            self.random_state = orig.random_state

            self.conv_sigma = orig.conv_sigma
            self.conv_resolution = orig.conv_resolution
            self.conv_gauss = orig.conv_gauss

            self.psf = orig.psf

            self.wave = orig.wave
            self.wave_edges = orig.wave_edges
            self.wave_lin = orig.wave_lin
            self.wave_log = orig.wave_log
            self.wave_bins = orig.wave_bins
            self.wave_resampler = orig.resampler
            
            self.norm = orig.norm
            self.norm_wave = orig.norm_wave

    def add_args(self, parser):
        parser.add_argument('--restframe', action='store_true', help='Convert to rest-frame.\n')
        parser.add_argument('--conv-sigma', type=float, help='Gaussian convolution sigma in AA.\n')
        parser.add_argument('--conv-resolution', type=float, help='Gaussian convolution to resolution.\n')
        parser.add_argument('--conv-gauss', type=str, help='Convolve using a Gauss PSF file.\n')

        parser.add_argument('--wave', type=float, nargs=2, default=None, help='Wavelength range.\n')
        parser.add_argument('--wave-bins', type=int, help='Number of wavelength bins.\n')
        parser.add_argument('--wave-lin', action='store_true', help='Linear wavelength binning.\n')
        parser.add_argument('--wave-log', action='store_true', help='Logarithmic wavelength binning.\n')
        parser.add_argument('--wave-resampler', type=str, default='rebin', choices=['interp', 'rebin'], help='Flux resampler.\n')

        parser.add_argument('--norm', type=str, default=None, help='Normalization method\n')
        parser.add_argument('--norm-wave', type=float, nargs=2, default=[4200, 6500], help='Normalization method\n')

    def init_from_args(self, config, args, sampler=None):
        self.restframe = self.get_arg('restframe', self.restframe, args)
            
        self.conv_sigma = self.get_arg('conv_sigma', self.conv_sigma, args)
        self.conv_resolution = self.get_arg('conv_resolution', self.conv_resolution, args)
        self.conv_gauss = self.get_arg('conv_gauss', self.conv_gauss, args)

        # If only wave limits are specified, the input grid will be truncated
        # If the number of bins are specified as well, resampling will be done
        self.wave = self.get_arg('wave', self.wave, args)
        self.wave_bins = self.get_arg('wave_bins', self.wave_bins, args)
        self.wave_lin = self.get_arg('wave_lin', self.wave_lin, args)
        self.wave_log = self.get_arg('wave_log', self.wave_log, args)

        resampler = self.get_arg('wave_resampler', None, args)
        if resampler == 'interp':
            self.wave_resampler = Interp1dResampler()
        elif resampler == 'rebin':
            self.wave_resampler = FluxConservingResampler()
        elif resampler is None:
            pass
        else:
            raise NotImplementedError()
        
        self.norm = self.get_arg('norm', self.norm, args)
        self.norm_wave = self.get_arg('norm_wave', self.norm_wave, args)

    def init_random_state(self, random_state):
        self.random_state = random_state

    def init_sampler_parameters(self, sampler):
        """
        Initialize additional axes (parameters) that will be sampled.
        """

        sampler.add_parameter(Parameter('z'))
        sampler.add_parameter(Parameter('rv'))
      
    def is_constant_wave(self):
        # Returns true when the wavelength grid is the same for all processed spectra
        return not self.restframe and self.redshift is None

    def get_wave(self, spec: Spectrum = None, **kwargs):
        if isinstance(self.wave, np.ndarray):
            return self.wave, self.wave_edges, None
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
            return self.wave, self.wave_edges, None
        elif self.wave_log:
            # Logarithmic binning with a constant grid
            edges = np.linspace(np.log10(self.wave[0]), np.log10(self.wave[1]), self.wave_bins + 1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            self.wave = 10 ** centers
            self.wave_edges = 10 ** np.stack([edges[:-1], edges[1:]])
            return self.wave, self.wave_edges, None
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
        self.run_step_resample(spec, **kwargs)
        self.run_step_normalize(spec, **kwargs)

    def run_step_restframe(self, spec: Spectrum, **kwargs):
        # If the spectrum is already redshifted, convert it to restframe
        spec.set_restframe()

    def run_step_redshift(self, spec: Spectrum, **kwargs):
        if self.is_arg('rv', kwargs):
            rv = self.get_arg('rv', None, kwargs)
            z = Physics.vel_to_z(rv)
        elif self.is_arg('z', kwargs):
            z = self.get_arg('z', None, kwargs)
        elif self.is_arg('redshift', kwargs):
            z = self.get_arg('redshift', None, kwargs)
        else:
            z = None

        if z is not None and not np.isnan(z) and z != 0.0:
            spec.set_redshift(z)

    def run_step_resample(self, spec, **kwargs):
        if self.wave is not None:
            wave, wave_edges, _ = self.get_wave()
            spec.apply_resampler(self.wave_resampler, wave, wave_edges)

    #region Convolution

    def init_psf(self, spec):

        # TODO: This is all wrong because the wavelength grid might change
        #       due to redshifting
        raise NotImplementedError()

        # Initialize PSF kernel, if not already

        # TODO: differentiate based on log-wave binning

        if self.psf is None:
            if self.conv_sigma is not None:
                self.psf = GaussPsf(sigma=self.conv_sigma)
            elif self.conv_resolution is not None:
                # TODO: create an interpolated gauss kernel to degrade resolution
                #       take resolution from the spectrum
                raise NotImplementedError()
            elif self.conv_gauss is not None:
                # Load tabulated kernel file from disk
                psf = GaussPsf(reuse_kernel=False)
                psf.load(self.conv_gauss, format='h5')

                # Calculate kernel sigma from target sigma
                if spec.resolution is not None:
                    input_dlam = psf.wave / spec.resolution
                    input_sigma = input_dlam / (2 * np.sqrt(2 * np.log(2)))
                    # Calculate kernel sigma
                    psf.sigma = np.sqrt(psf.sigma**2 - input_sigma**2)
                    psf.init_ip()

                # Precompute PCA for faster convolution
                # TODO: how do we know that wave is constant?
                size = psf.get_optimal_size(spec.wave)
                self.psf = PcaPsf.from_psf(psf, spec.wave, size=size)

    def run_step_convolution(self, spec, **kwargs):
        if self.psf is not None:
            self.init_psf(spec)

            # Convolve with a PSF
            spec.convolve_psf(self.psf)

    #endregion

    def run_step_normalize(self, spec, **kwargs):
        # TODO: rewrite these to classes?
        # Normalization in wavelength range overrides previous normalization
        wave = self.norm_wave or (spec.wave[0], spec.wave[-1])
        if self.norm == 'none':
            pass
        elif self.norm == 'median':
            spec.normalize_in([wave], np.median, value=0.5)
        elif self.norm == 'mean':
            spec.normalize_in([wave], np.mean, value=0.5)
        elif self.norm == 'integral':
            # TODO: This is not integral flux but sum of flux only!
            #       It only works when binning is linear       
            spec.normalize_in([wave], np.sum, value=1.0)
        elif self.norm is not None:
            raise NotImplementedError()

    def set_if_valid_param(self, spec, name, **kwargs):
        if name in kwargs and kwargs[name] is not None and not np.isnan(kwargs[name]):
            setattr(spec, name, kwargs[name])
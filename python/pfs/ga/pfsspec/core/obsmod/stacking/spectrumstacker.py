import numpy as np

from ...util.copy import *
from ...util.args import *
from ... import Spectrum
from ... import Physics
from ..resampling import Interp1dResampler, FluxConservingResampler, Binning

class SpectrumStacker():
    # Base class for stacking multiple observations

    # For now, we assume that the spectra are in the same velocity frame
    # but the wavelength grids may be different.

    # TODO: Convolve spectra to a common resolution
    # TODO: Move redshift correction here

    def __init__(self, trace=None, orig=None):
        
        if not isinstance(orig, SpectrumStacker):
            self.trace = trace

            self.spectrum_type = Spectrum

            self.binning = None             # Binning: lin or log
            self.nbins = None               # Number of new bins
            self.binsize = None             # Spectral bin size (lambda or log lambda) after resampling

            self.normalize_cont = False     # Normalize the continuum, if continuum is present
            self.apply_flux_corr = False    # Use flux correction, if available

            self.mask_no_data_bit = 1            # Bit to set in the mask for no data
            self.mask_exclude_bits = None   # List of mask bits to exclude from coadding

            # self.resampler = FluxConservingResampler()
            self.resampler = Interp1dResampler()

        else:
            self.trace = trace if trace is not None else orig.trace

            self.spectrum_type = orig.spectrum_type

            self.binning = orig.binning
            self.nbins = orig.nbins
            self.binsize = orig.binsize

            self.normalize_cont = orig.normalize_cont
            self.apply_flux_corr = orig.apply_flux_corr

            self.mask_no_data_bit = orig.mask_no_data_bit
            self.mask_exclude_bits = orig.mask_exclude_bits

            self.resampler = orig.resampler

    def add_args(self, config, parser):
        super().add_args(config, parser)
    
    def init_from_args(self, script, config, args):
        if self.trace is not None:
            self.trace.init_from_args(script, config, args)

        self.binning = get_arg('binning', self.binning, args)
        self.nbins = get_arg('nbins', self.nbins, args)
        self.binsize = get_arg('binsize', self.binsize, args)

        self.normalize_cont = get_arg('normalize_cont', self.normalize_cont, args)
        self.apply_flux_corr = get_arg('apply_flux_corr', self.apply_flux_corr, args)

    def stack(self, 
              spectra: list,
              target_wave=None, target_wave_edges=None,
              v_corr: list = None):
        
        """
        Resample and stack the spectra to a common grid

        This function destroys the input spectra so only pass a copy of them
        to it if you want to keep the original flux.

        Parameters
        ----------
        spectra : list
            List of Spectrum objects
        target_wave : array
            Target wavelength grid, optional. If not specified, the wavelength grid is determined
            from the input spectra.
        target_wave_edges : array
            Target wavelength bin edges, optional. If not specified, the wavelength bin edges are
            determined from the input spectra.
        v_corr : list
            List of velocity corrections for the input spectra, optional. If specified, spectra are
            shifted by the velocity correction before stacking.
        """

        # Call the trace hook to plot the input spectra
        if self.trace is not None:
            self.trace.on_coadd_start_stack(
                { 'any': spectra },
                self.mask_no_data_bit,
                self.mask_exclude_bits
            )

        # Common wavelength grid
        redshifts = self.__get_redshifts(spectra, v_corr=v_corr)
        target_wave, target_wave_edges = self.__get_target_wave(
            spectra,
            target_wave=target_wave, target_wave_edges=target_wave_edges,
            redshifts=redshifts)

        # Resample every spectrum as sum up flux
        # TODO: this needs a bit more work:
        #       - error is now interpolated linearly, replace
        #         it with proper quadrature of the integrated pixels
        #       - calculate full error covariance matrix


        # TODO: separate resampling and stacking into two pipeline steps?
        # TODO: how to stack up covariance?

        # Create vectors for observed quantities but never the models
        shape = target_wave.shape
        dtype = spectra[0].flux.dtype
        mask_dtype = spectra[0].mask.dtype

        stacked_flux = np.zeros(shape, dtype=dtype)
        stacked_flux_err = np.zeros(shape, dtype=dtype)
        stacked_mask = np.zeros(shape, dtype=mask_dtype)
        stacked_weight = np.zeros(shape, dtype=dtype)
        stacked_flux_sky = np.zeros(shape, dtype=dtype)

        is_flux_calibrated = True
        has_sky = True
        mask_flags = None

        for i, spec in enumerate(spectra):
            is_flux_calibrated &= spec.is_flux_calibrated
            has_sky &= spec.flux_sky is not None
            mask_flags = mask_flags if mask_flags is not None else spec.mask_flags

            # TODO: Do we need to rescale the flux when correcting for the Doppler shift?
            # TODO: Move the redshifting to the resampler class
            if v_corr is not None:
                z = redshifts[i]
                wave = spec.wave * (1 + z)
                wave_edges = spec.wave_edges * (1 + z)
            else:
                wave = spec.wave
                wave_edges = spec.wave_edges

            # Apply the flux correction to the observed spectra. Note than when fitting the
            # templates, the flux correction is a multiplier of the templates, hence we have
            # to divide the observed spectra with it.
            if self.apply_flux_corr and spec.flux_corr is not None:
                spec.multiply(1.0 / spec.flux_corr)

            if self.normalize_cont and spec.cont is not None:
                spec.multiply(1.0 / spec.cont)

            # Mask the target wavelength range to the range of the input spectrum
            # so that no extrapolation happens during resampling. To resample the flux
            # we use the wave edges but the error is resampled using the wave centers
            # so we need to make sure both are properly masked.
            wave_mask = (wave[0] <= target_wave) & (target_wave <= wave[-1]) & \
                        (wave_edges[0] <= target_wave_edges[:-1]) & (target_wave_edges[1:] <= wave_edges[-1])
            wave_edges_mask = (wave_edges[0] <= target_wave_edges) & (target_wave_edges <= wave_edges[-1])
            wave_edges_mask[1:] &= wave_mask
            wave_edges_mask[:-1] &= wave_mask
            wave_mask &= wave_edges_mask[1:]
            wave_mask &= wave_edges_mask[:-1]

            # wave_mask = (wave[0] <= target_wave) & (target_wave <= wave[-1]) & \
            #             (wave_edges[0] <= target_wave_edges[:-1]) & (target_wave_edges[1:] <= wave_edges[-1])
            # wave_edges_mask = (wave_edges[0] <= target_wave_edges) & (target_wave_edges <= wave_edges[-1])

            flux, flux_err, resampler_mask = self.resampler.resample_flux(
                wave, wave_edges, spec.flux, spec.flux_err,
                target_wave=target_wave[wave_mask],
                target_wave_edges=target_wave_edges[wave_edges_mask])
            
            # TODO: this assumes that the mask bits are the same for every spectrum but this is not
            #       necessarily the case
            mask = self.resampler.resample_mask(wave, wave_edges, spec.mask,
                                                       target_wave=target_wave[wave_mask],
                                                       target_wave_edges=target_wave_edges[wave_edges_mask])
            
            if has_sky:
                flux_sky, _, _ = self.resampler.resample_flux(
                    wave, wave_edges, spec.flux_sky, spec.flux_err,
                    target_wave=target_wave[wave_mask],
                    target_wave_edges=target_wave_edges[wave_edges_mask])
            
            # Calculate the weight vector for each pixel of the current spectrum
            # TODO: use exposure time or we're happy with weighting by relative error?
            # TODO: verify if this is correct in case of normalized spectra
            weight = 1.0 / flux_err ** 2
            
            # Give zero weight to masked pixels
            if self.mask_exclude_bits is not None:
                weight[(mask & self.mask_exclude_bits) != 0] = 0
            
            # TODO: add trace hook

            # Stack up vectors. Errors are assumed to be absolute and summed up in quadrature.
            stacked_flux[wave_mask] = stacked_flux[wave_mask] + weight * np.where(resampler_mask, flux, 0.0)
            stacked_weight[wave_mask] = stacked_weight[wave_mask] + np.where(resampler_mask, weight, 0.0)
            if has_sky:
                stacked_flux_sky[wave_mask] = stacked_flux_sky[wave_mask] + np.where(resampler_mask, flux_sky, 0.0)
            
            # Only consider that part of the mask where the flux could be calculated
            # Resampler_mask is True for the pixels where the flux could be calculated by the interpolator
            if mask.dtype == bool:
                stacked_mask[wave_mask] = stacked_mask[wave_mask] | np.where(resampler_mask, mask, False)
            else:
                stacked_mask[wave_mask] = stacked_mask[wave_mask] | np.where(resampler_mask, mask, self.mask_no_data_bit)

        # TODO: add trace hook

        stacked_flux = stacked_flux / stacked_weight
        stacked_flux_err = np.sqrt(1 / stacked_weight)
        stacked_flux_err[~np.isfinite(stacked_flux_err)] = 0.0
        if has_sky:
            stacked_flux_sky = stacked_flux_sky / stacked_weight

        # Mask out bins where the weight is zero
        stacked_mask = np.where(stacked_weight == 0, stacked_mask | self.mask_no_data_bit, stacked_mask)

        # Create a new Spectrum object
        spec = self.spectrum_type()

        # TODO: 
        if self.binning == 'log':
            spec.is_wave_regular = True
            spec.is_wave_lin = False
            spec.is_wave_log = True
        elif self.binning == 'lin':
            spec.is_wave_regular = True
            spec.is_wave_lin = True
            spec.is_wave_log = False
        else:
            raise NotImplementedError()

        spec.is_flux_calibrated = is_flux_calibrated
        spec.mask_flags = mask_flags

        spec.wave = target_wave
        spec.wave_edges = target_wave_edges
        spec.flux = stacked_flux
        spec.flux_err = stacked_flux_err
        spec.mask = stacked_mask
        if has_sky:
            spec.flux_sky = stacked_flux_sky

        # TODO: sky? covar? covar2? - these are required for a valid PfsFiberArray
        # TODO: these should go into the stacker class
        spec.covar = np.zeros((3,) + spec.wave.shape)
        spec.covar[1, :] = spec.flux_err**2
        spec.covar2 = np.zeros((1, 1), dtype=np.float32)

        if self.trace is not None:
            self.trace.on_coadd_finish_stack(
                { 'any': [spec] },
                self.mask_no_data_bit,
                self.mask_exclude_bits)

        return spec

    def merge(self, spectra: dict):
        
        """
        Merge spectra from multiple arms into a single spectrum.

        Parameters
        ----------
        spectra : dict
            Dict of Spectrum objects
        """

        # TODO: we currently assume that each arm has only one spectrum
        assert all([ len(spectra[arm]) == 1 for arm in spectra ])

        # Sort arms by wavelength
        sorted_arms = sorted(spectra.keys(), key=lambda arm: np.min(spectra[arm][0].wave))

        def concatenate_vectors(vfunc):
            if all(vfunc(spectra[arm][0]) is not None for arm in sorted_arms):
                return np.concatenate([ vfunc(spectra[arm][0]) for arm in sorted_arms ])
            else:
                return None

        # Create a new Spectrum object
        spec = self.spectrum_type()

        # Concatenate spectra from different arms
        # TODO: figure out how to iterate over all vectors in the Spectrum class
        spec.wave = concatenate_vectors(lambda s: s.wave)
        spec.wave_edges = concatenate_vectors(lambda s: s.wave_edges)
        spec.flux = concatenate_vectors(lambda s: s.flux)
        spec.flux_model = concatenate_vectors(lambda s: s.flux_model)
        spec.flux_err = concatenate_vectors(lambda s: s.flux_err)
        spec.flux_sky = concatenate_vectors(lambda s: s.flux_sky)
        spec.flux_corr = concatenate_vectors(lambda s: s.flux_corr)
        spec.mask = concatenate_vectors(lambda s: s.mask)
        spec.weight = concatenate_vectors(lambda s: s.weight)
        spec.cont = concatenate_vectors(lambda s: s.cont)
        spec.cont_fit = concatenate_vectors(lambda s: s.cont_fit)

        if self.trace is not None:
            self.trace.on_coadd_finish_merged(spec)

        return spec

    def __get_target_wave(self, spectra, target_wave=None, target_wave_edges=None, redshifts=None):
        # Get the common wavelength grid for the spectra

        # If not specified, determine the wavelength bin edges
        if target_wave is None and target_wave_edges is None:
            wmin, wmax = self.__get_wave_min_max(spectra, redshifts)
            # If `binsize` is specified, round `wmin` and `wmax` to the closest integer multiple of `binsize`
            if self.binsize is not None:
                wmin, wmax = Binning.round_wave_min_max(wmin, wmax, self.binsize, binning=self.binning)

            target_wave, target_wave_edges = Binning.generate_wave_bins(wmin, wmax, nbins=self.nbins, binsize=self.binsize, binning=self.binning)
        elif target_wave_edges is None:
            # TODO: Do not automatically assume linear binning of the input spectra
            target_wave_edges = Binning.find_wave_edges(target_wave, binning='lin')
            
        return target_wave, target_wave_edges
    
    def __get_redshifts(self, spectra, v_corr=None):
        # Calculate the redshifts from the velocity corrections
        if v_corr is not None:
            return [ Physics.vel_to_z(v) for v in v_corr ]
        else:
            return [ 0.0 for s in spectra ]
    
    def __get_wave_min_max(self, spectra: list, redshifts: list):
        # Determine minimum and maximum wavelengths
        wmin, wmax = None, None
        for s, z in zip(spectra, redshifts):
            if s.wave_edges is None:
                we = Binning.find_wave_edges(s.wave) * (1 + z)
            else:
                we = s.wave_edges * (1 + z)

            if wmin is None or we[0] < wmin:
                wmin = we[0]
            if wmax is None or wmax < we[-1]:
                wmax = we[-1]

        return wmin, wmax
    
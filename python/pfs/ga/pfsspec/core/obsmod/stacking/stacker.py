import numpy as np

from pfs.ga.pfsspec.core.util.copy import *
from pfs.ga.pfsspec.core.util.args import *
from ... import Physics
from ..resampling import FluxConservingResampler, Binning

class Stacker():
    # Base class for stacking multiple observations

    # For now, we assume that the spectra are in the same velocity frame
    # but the wavelength grids may be different.

    # TODO: Convolve spectra to a common resolution
    # TODO: Move redshift correction here

    def __init__(self, trace=None, orig=None):
        
        if not isinstance(orig, Stacker):
            self.trace = trace

            self.binning = None             # Binning: lin or log
            self.nbins = None               # Number of new bins
            self.binsize = None             # Spectral bin size (lambda or log lambda) after resampling

            self.resampler = FluxConservingResampler()
        else:
            self.trace = trace if trace is not None else orig.trace

            self.binning = orig.binning
            self.nbins = orig.nbins
            self.binsize = orig.binsize

            self.resampler = orig.resampler

    def add_args(self, config, parser):
        super().add_args(config, parser)
    
    def init_from_args(self, script, config, args):
        if self.trace is not None:
            self.trace.init_from_args(script, config, args)

        self.binning = get_arg('binning', self.binning, args)
        self.nbins = get_arg('nbins', self.nbins, args)
        self.binsize = get_arg('binsize', self.binsize, args)

    def stack(self, 
              spectra: list,
              target_wave=None, target_wave_edges=None,
              v_corr: list = None, weights: list = None,
              flux_corr: list = None):
        # Resample and stack the spectra to a common grid

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
        #       - mask is taken to be the nearest neighbor, instead
        #         calculate the binary OR of the integrated pixels

        stacked_flux = np.zeros(target_wave.shape, dtype=spectra[0].flux.dtype)
        stacked_error = np.zeros(target_wave.shape, dtype=spectra[0].flux.dtype)
        stacked_mask = np.zeros(target_wave.shape, dtype=spectra[0].mask.dtype)
        stacked_weight = np.zeros(target_wave.shape, dtype=spectra[0].flux.dtype)

        for i, spec in enumerate(spectra):
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
            if flux_corr is not None:
                corr = flux_corr[i]
                flux = spec.flux / corr
                error = spec.flux_err / corr
            else:
                flux = spec.flux
                error = spec.flux_err

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

            flux, error, resampler_mask = self.resampler.resample_flux(wave, wave_edges, flux, error,
                                                target_wave=target_wave[wave_mask],
                                                target_wave_edges=target_wave_edges[wave_edges_mask])
            
            # TODO: this assumes that the mask bits are the same for every spectrum but this is not
            #       necessarily the case
            mask = self.resampler.resample_mask(wave, wave_edges, spec.mask,
                                                target_wave=target_wave[wave_mask],
                                                target_wave_edges=target_wave_edges[wave_edges_mask])
            
            # TODO: add trace hook

            # Stack up vectors. Errors are assumed to be absolute and summed up in quadrature.
            stacked_flux[wave_mask] = stacked_flux[wave_mask] + np.where(resampler_mask, flux, 0.0)
            stacked_error[wave_mask] = stacked_error[wave_mask] + np.where(resampler_mask, error ** 2, 0.0)
            stacked_weight[wave_mask] = stacked_weight[wave_mask] + np.where(resampler_mask, weights[i] if weights is not None else 1.0, 0.0)
            stacked_mask[wave_mask] = stacked_mask[wave_mask] | np.where(resampler_mask, mask, False)
            
        # TODO: add trace hook

        stacked_flux /= stacked_weight
        stacked_error = np.sqrt(stacked_error)

        return target_wave, target_wave_edges, stacked_flux, stacked_error, stacked_weight, stacked_mask

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
    
import numpy as np

from ...util.copy import *
from ...util.args import *
from ... import Spectrum

class SpectrumMerger():
    # Class for merging spectra from multiple arms

    # TODO: the current implementation assumes that the spectrograph arms
    #       do not overlap
    def __init__(self, trace=None, orig=None):
        
        if not isinstance(orig, SpectrumMerger):
            self.trace = trace

            self.spectrum_type = Spectrum
        else:
            self.trace = trace if trace is not None else orig.trace

            self.spectrum_type = orig.spectrum_type

    def add_args(self, config, parser):
        super().add_args(config, parser)
    
    def init_from_args(self, script, config, args):
        if self.trace is not None:
            self.trace.init_from_args(script, config, args)

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

        return spec

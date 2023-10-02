import numpy as np

from .snr import Snr

class StoehrSnr(Snr):
    def __init__(self, shift=2, binning=1.0, orig=None):
        """
        Estimate the S/N using Stoehr et al ADASS 2008
           
        signal = median(flux(i))
        noise = 1.482602 / sqrt(6.0) *
        median(abs(2 * flux(i) - flux(i-2) - flux(i+2)))
        DER_SNR = signal / noise
        """

        # NOTE: This doesn't really work with the ETC noise model because
        #       of the maximum filter involved in the the sky subtraction
        #       error estimate.
        
        super().__init__(binning, orig=orig)

        if not isinstance(orig, StoehrSnr):
            self.shift = shift if shift is None else 2
        else:
            self.shift = shift if shift is not None else orig.q

    def get_snr(self, value, sigma=None):
        raise NotImplementedError()
        # TODO: update to handle lists of values/sigmas and mask

        s1 = np.median(value)
        s2 = np.abs(2 * value[self.shift:-self.shift] - value[:-2 * self.shift] - value[2 * self.shift:])
        n1 = 1.482602 / np.sqrt(6.0) * np.median(s2)
        snr = s1 / n1 * np.sqrt(self.binning)
        return snr
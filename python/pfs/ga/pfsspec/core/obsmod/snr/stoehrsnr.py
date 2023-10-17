import numpy as np

from .snr import Snr

class StoehrSnr(Snr):
    def __init__(self, binning=1.0, squared=None, shift=2, orig=None):
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
        
        super().__init__(binning=binning, squared=squared, orig=orig)

        if not isinstance(orig, StoehrSnr):
            self.shift = shift if shift is None else 2
        else:
            self.shift = shift if shift is not None else orig.shift

    def get_snr(self, value, sigma, mask):
        # TODO: Handle cases when the mask would cause taking the difference of non-adjacent bins

        signal = np.median(value[mask])
        df = np.abs(2 * value[mask][self.shift:-self.shift] - value[mask][:-2 * self.shift] - value[mask][2 * self.shift:])
        noise = 1.482602 / np.sqrt(6.0) * np.median(df)
        snr = signal / noise * np.sqrt(self.binning)
        return snr
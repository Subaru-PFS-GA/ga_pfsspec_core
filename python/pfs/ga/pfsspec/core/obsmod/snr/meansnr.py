import numpy as np

from .snr import Snr

class MeanSnr(Snr):
    """
    Estimate SNR as the mean of value/sigma.
    """

    def __init__(self, binning=1.0, squared=None, orig=None):
        super().__init__(binning=binning, squared=squared, orig=orig)

        if not isinstance(orig, MeanSnr):
            pass
        else:
            pass

    def get_snr_impl(self, value, sigma, mask):
        return np.mean(value[mask] / sigma[mask]) * np.sqrt(self.binning)

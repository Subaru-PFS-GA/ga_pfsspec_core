import numpy as np

from .snr import Snr

class MedianSnr(Snr):
    """
    Estimate SNR as the median of value/sigma.
    """

    def __init__(self, binning=1.0, squared=None, orig=None):
        super().__init__(binning=binning, squared=squared, orig=orig)

        if not isinstance(orig, MedianSnr):
            pass
        else:
            pass

    def get_snr_impl(self, value, sigma, mask):
        m =  np.median(value[mask] / sigma[mask]) * np.sqrt(self.binning)
        return m
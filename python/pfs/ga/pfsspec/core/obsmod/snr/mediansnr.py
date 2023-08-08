import numpy as np

from .snr import Snr

class MedianSnr(Snr):
    """
    Estimate SNR as the median of value/sigma.
    """

    def __init__(self, binning=1.0, orig=None):
        super().__init__(binning=binning, orig=orig)

        if not isinstance(orig, MedianSnr):
            pass
        else:
            pass

    def get_snr_impl(self, value, sigma, mask):
        return np.median(value[mask] / sigma[mask])

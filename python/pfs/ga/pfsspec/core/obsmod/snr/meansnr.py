import numpy as np

from .snr import Snr

class MeanSnr(Snr):
    """
    Estimate SNR as the mean of value/sigma.
    """

    def __init__(self, binning=1.0, orig=None):
        super().__init__(binning=binning, orig=orig)

        if not isinstance(orig, MeanSnr):
            pass
        else:
            pass

    def get_snr(self, value, sigma=None):
        if sigma is None:
            return np.nan
        else:
            mask = (sigma > 0)
            snr = np.mean(value[mask] / sigma[mask]) * np.sqrt(self.binning)
            return snr

    
import numpy as np

class Snr():
    """
    When overriden in derived classes, implements a signal to noise ratio
    calculation algorithm.
    """

    def __init__(self, binning=None, orig=None):
        if not isinstance(orig, Snr):
            self.binning = binning if binning is not None else 1.0
        else:
            self.binning = binning if binning is not None else orig.binning

    def get_snr(self, values, sigmas=None, masks=None):
        # Allow passing lists of values just in case the SNR is to be
        # calculated for multiple arms

        if isinstance(values, np.ndarray):
            values = [ values ]

        if isinstance(sigmas, np.ndarray):
            sigmas = [ sigmas ]

        if isinstance(masks, np.ndarray):
            masks = [ masks ]

        if sigmas is None:
            return np.nan
        else:
            v = np.concatenate(values)
            s = np.concatenate(sigmas)
            if masks is not None:
                m = np.concatenate(masks)
                mask = m & (s > 0.0)
            else:                
                mask = (s > 0.0)

            mask = mask & ~(np.isnan(v) | np.isnan(s))

            snr = self.get_snr_impl(v, s, mask)
            return snr
        
    def get_snr_impl(self, value, sigma, mask):
        raise NotImplementedError()
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

    def get_snr(self, value, sigma=None):
        raise NotImplementedError()

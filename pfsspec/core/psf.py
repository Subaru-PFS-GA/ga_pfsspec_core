from .pfsobject import PfsObject

class Psf(PfsObject):
    """
    When implemented in derived classes, it calculates the points spread
    function kernel.
    """

    def __init__(self):
        super(Psf, self).__init__()

    def get_kernel(self, lam, dwave=None, size=None):
        raise NotImplementedError()
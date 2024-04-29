from ...pfsobject import PfsObject

class FluxCorrection(PfsObject):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

    def get_param_count(self):
        raise NotImplementedError()

    def get_basis_callable(self):
        """
        Return a callable that evaluates the basis functions over a
        wavelength grid.
        """
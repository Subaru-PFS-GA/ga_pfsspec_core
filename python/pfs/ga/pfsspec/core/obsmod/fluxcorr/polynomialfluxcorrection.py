import numpy as np

from .fluxcorrectionmodel import FluxCorrectionModel

class PolynomialFluxCorrection(FluxCorrectionModel):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, PolynomialFluxCorrection):
            self.function = 'chebyshev'
            self.degree = 5
            self.wlim = (3000.0, 12000.0)       # Wavelength normalization interval
        else:
            self.function = orig.function
            self.degree = orig.degree
            self.wlim = orig.wlim

    def get_param_count(self):
        """
        Return the number of model parameters
        """
        return self.degree

    def get_basis_callable(self):
        """
        Return a callable that evaluates the basis functions over a
        wavelength grid.
        """

        def fun(wave):
            if self.function == 'chebyshev':
                t = np.polynomial.Chebyshev
            else:
                raise NotImplementedError()

            normwave = (wave - self.wlim[0]) / (self.wlim[1] - self.wlim[0]) * 2 - 1
            polys = np.empty((wave.shape[0], self.degree))

            coeffs = np.eye(self.degree)
            for i in range(self.degree):
                polys[:, i] = t(coeffs[i])(normwave)

            return polys

        return fun


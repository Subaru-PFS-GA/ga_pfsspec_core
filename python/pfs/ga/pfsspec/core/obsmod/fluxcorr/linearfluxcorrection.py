from .fluxcorrection import FluxCorrection

class LinearFluxCorrection(FluxCorrection):
    """
    Base class for flux correction models where the linear coefficients
    are the only unknown. This includes piecewise basis functions etc.
    """

    pass

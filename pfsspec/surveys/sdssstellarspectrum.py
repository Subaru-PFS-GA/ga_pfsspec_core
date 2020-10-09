from pfsspec.obsmod.stellarspectrum import StellarSpectrum
from pfsspec.surveys.sdssspectrum import SdssSpectrum

class SdssStellarSpectrum(StellarSpectrum, SdssSpectrum):
    def __init__(self, orig=None):
        StellarSpectrum.__init__(self, orig=orig)
        SdssSpectrum.__init__(self, orig=orig)

    def get_param_names(self):
        params = SdssSpectrum.get_param_names(self)
        params += StellarSpectrum.get_param_names(self)
        return params

    def print_info(self):
        SdssSpectrum.print_info(self)
        StellarSpectrum.print_info(self)
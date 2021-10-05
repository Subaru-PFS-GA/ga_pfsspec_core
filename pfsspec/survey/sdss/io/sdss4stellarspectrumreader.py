from pfsspec.surveys.sdss4spectrumreader import Sdss4SpectrumReader
from pfsspec.surveys.sdssseguespectrum import SdssSegueSpectrum

class Sdss4StellarSpectrumReader(Sdss4SpectrumReader):
    def __init__(self, orig=None):
        super(Sdss4StellarSpectrumReader, self).__init__(orig=orig)

        if isinstance(orig, Sdss4StellarSpectrumReader):
            pass
        else:
            pass

    def create_spectrum(self, hdus):
        return SdssSegueSpectrum()

    def read_header(self, hdus, row, spec):
        super(Sdss4StellarSpectrumReader, self).read_header(hdus, row, spec)

        spec.T_eff = row['T_eff']
        spec.Fe_H = row['Fe_H']
        spec.log_g = row['log_g']
        spec.a_Fe = row['a_Fe']
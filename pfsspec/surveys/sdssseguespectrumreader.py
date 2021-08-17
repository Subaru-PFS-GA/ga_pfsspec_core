from pfsspec.surveys.sdssspectrumreader import SdssSpectrumReader
from pfsspec.surveys.sdssseguespectrum import SdssSegueSpectrum

class SdssSegueSpectrumReader(SdssSpectrumReader):
    def __init__(self, orig=None):
        super(SdssSegueSpectrumReader, self).__init__(orig=orig)

        if isinstance(orig, SdssSegueSpectrumReader):
            pass
        else:
            pass

    def create_spectrum(self, hdus):
        return SdssSegueSpectrum()

    def read_header(self, hdus, row, spec):
        super(SdssSegueSpectrumReader, self).read_header(hdus, row, spec)

        spec.T_eff = row['T_eff']
        spec.Fe_H = row['Fe_H']
        spec.log_g = row['log_g']
        spec.a_Fe = row['a_Fe']
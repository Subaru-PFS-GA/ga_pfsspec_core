import sys

from pfsspec.data.spectrumreader import SpectrumReader
from pfsspec.surveys.survey import Survey
from pfsspec.parallel import SmartParallel

class SurveySpectrumReader(SpectrumReader):
    def __init__(self, verbose, parallel):
        super(SurveySpectrumReader, self).__init__(verbose=verbose, parallel=parallel)

    def load_spectrum(self, index, row):
        raise NotImplementedError()

    def load_spectrum_wrapper(self, ix_row):
        index, row = ix_row
        try:
            return self.load_spectrum(index, row)
        except Exception as e:
            print(e, file=sys.stderr)   # TODO
            return None

    def load_survey(self, params):
        survey = Survey()
        survey.params = params
        survey.spectra = []

        rows = [(index, row) for index, row in params.iterrows()]
        with SmartParallel(verbose=self.verbose, parallel=self.parallel) as p:
            survey.spectra = [r for r in p.map(self.load_spectrum_wrapper, rows)]
        # Parallal will likely shuffle the spectra    
        survey.spectra.sort(key=lambda s: s.index)
        return survey
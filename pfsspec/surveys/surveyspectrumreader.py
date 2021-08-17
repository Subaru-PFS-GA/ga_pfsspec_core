import sys

from pfsspec.data.spectrumreader import SpectrumReader
from pfsspec.surveys.survey import Survey
from pfsspec.util.parallel import SmartParallel

class SurveySpectrumReader(SpectrumReader):
    def __init__(self, verbose=False, parallel=False, threads=None):
        super(SurveySpectrumReader, self).__init__()

        self.verbose = verbose
        self.parallel = parallel
        self.threads = threads

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

        # TODO: this requires loading everything into memory
        #       consider redesigning the dataset builder to read the spectra on
        #       multiple threads instead

        rows = [(index, row) for index, row in params.iterrows()]
        with SmartParallel(verbose=self.verbose, parallel=self.parallel, threads=self.threads) as p:
            survey.spectra = [r for r in p.map(self.load_spectrum_wrapper, rows)]
        # Parallal will likely shuffle the spectra    
        survey.spectra.sort(key=lambda s: s.index)
        return survey
import os
import sys

import pfsspec.util as util
from pfsspec.util.parallel import SmartParallel
from pfsspec.data.importer import Importer
from pfsspec.data.survey import Survey

class SurveyReader(Importer):
    """
    Implements function to read spectra of a survey.
    """

    def __init__(self, orig=None):
        super(SurveyReader, self).__init__(orig=orig)

        if isinstance(orig, SurveyReader):
            self.verbose = orig.verbose
            self.parallel = orig.parallel
            self.threads = orig.threads
            self.resume = orig.resume
            self.top = orig.top
            self.outdir = orig.outdir
        else:
            self.verbose = True
            self.parallel = True
            self.threads = None
            self.resume = False
            self.top = None
            self.outdir = None

    def add_args(self, parser):
        parser.add_argument('--top', type=int, help='Stop after this many items.\n')

    def init_from_args(self, args):
        super(SurveyReader, self).init_from_args(args)

        self.threads = self.get_arg('threads', self.threads, args)
        self.parallel = self.threads is None or self.threads > 1
        if not self.parallel:
            self.logger.info('Survey reader running in sequential mode.')
        self.top = self.get_arg('top', self.top, args)

    def open_data(self, indir, outdir):
        fn = os.path.join(outdir, 'spectra.dat')

        self.outdir = outdir
        self.survey = self.create_survey()
        self.survey.filename = fn
        self.survey.fileformat = 'pickle'

    def save_data(self):
        self.survey.save()

    def create_survey(self):
        raise NotImplementedError()

    def create_spectrum_reader(self):
        raise NotImplementedError()

    def load_spectrum(self, index, row):
        return self.reader.load_spectrum(index, row)

    def process_item(self, ix_row):
        index, row = ix_row
        try:
            return self.load_spectrum(index, row)
        except Exception as e:
            self.logger.error(e)
            return None

    def store_item(self, ix_row, spec):
        # TODO: When optimizing memory use to process larger surveys,
        #       implement this function similarly to DatasetBuilder.store_item
        raise NotImplementedError()

    def load_survey(self, params):
        self.survey.params = params
        self.survey.spectra = []

        # TODO: this requires loading everything into memory
        #       consider redesigning the storage and read the spectra on
        #       multiple threads instead

        rows = [(index, row) for index, row in params.iterrows()]
        with SmartParallel(verbose=self.verbose, parallel=self.parallel, threads=self.threads) as p:
            self.survey.spectra = [r for r in p.map(self.process_item, rows)]
        
        # In case errors happened we get Nones
        self.survey.spectra = list(filter(lambda s: s is not None, self.survey.spectra))
        
        # Parallel will likely shuffle the spectra    
        self.survey.spectra.sort(key=lambda s: s.index)

    def run(self):
        raise NotImplementedError()

    def execute_notebooks(self, script):
        pass
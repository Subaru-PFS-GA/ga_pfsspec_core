import os
import logging

from pfsspec.parallel import SmartParallel
from pfsspec.data.spectrumwriter import SpectrumWriter

class DatasetSpectrumWriter(SpectrumWriter):
    def __init__(self, dataset=None):
        super(DatasetSpectrumWriter, self).__init__()

        self.outdir = None
        self.dataset = dataset
        self.pipeline = None

    def get_filename(self, spec):
        return os.path.combine(self.outdir, 'spec_{}.spec'.format(spec.id))

    def process_item(self, i):
        spec = self.dataset.get_spectrum(i)
        if self.pipeline is not None:
            self.pipeline.run(spec)

        file = self.get_filename(spec)
        self.write(file, spec)

    def write_all(self):
        rng = range(self.dataset.get_count())

        k = 0
        with SmartParallel(verbose=True, parallel=True) as p:
            for r in p.map(self.process_item, rng):
                k += 1

        logging.info('{} files written.'.format(k))
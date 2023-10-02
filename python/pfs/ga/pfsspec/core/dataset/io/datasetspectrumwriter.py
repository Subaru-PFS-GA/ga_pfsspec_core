import os
import logging

from pfs.ga.pfsspec.util.parallel import SmartParallel
from pfs.ga.pfsspec.data.spectrumwriter import SpectrumWriter

class DatasetSpectrumWriter(SpectrumWriter):
    # TODO: consider making DatasetSpectrumWriter a new branch of classes
    #       instead of inheriting from spectrum writer and make spectrumwriter
    #       a property instead so different types can be mixed and matched, just
    #       like with spectrumreader
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

        # TODO: set number of threads here somewhere.
        raise NotImplementedError()

        k = 0
        with SmartParallel(verbose=True, parallel=True) as p:
            for r in p.map(self.process_item, rng):
                k += 1

        self.logger.info('{} files written.'.format(k))
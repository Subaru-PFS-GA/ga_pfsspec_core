from pfsspec.data.datasetbuilder import DatasetBuilder

class SdssDatasetBuilder(DatasetBuilder):
    def __init__(self, orig=None):
        super(SdssDatasetBuilder, self).__init__(orig)
        if orig is not None:
            self.survey = orig.survey
        else:
            self.survey = None

    def get_spectrum_count(self):
        return len(self.survey.spectra)

    def get_wave_count(self):
        return self.pipeline.rebin.shape[0]

    def process_item(self, i):
        spec = self.survey.spectra[i]
        self.pipeline.run(spec)
        return spec

    def build(self):
        super(SdssDatasetBuilder, self).build()
        self.dataset.wave[:] = self.pipeline.rebin
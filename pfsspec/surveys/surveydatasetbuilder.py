from pfsspec.data.datasetbuilder import DatasetBuilder

class SurveyDatasetBuilder(DatasetBuilder):
    def __init__(self, orig=None, random_seed=None):
        super(SurveyDatasetBuilder, self).__init__(orig=orig, random_seed=random_seed)
        
        if orig is not None:
            self.survey = orig.survey
        else:
            self.survey = None

    def load_survey(self, filename):
        self.survey.load(filename)
        self.params = self.survey.params

    def get_spectrum_count(self):
        return len(self.survey.spectra)

    def get_wave_count(self):
        return self.pipeline.get_wave_count()

    def process_item(self, i):
        params = self.get_params(i)
        spec = self.survey.spectra[i]
        spec.set_params(params)
        self.pipeline.run(spec, **params)

        # TODO: use mask from spectra
        raise NotImplementedError()

        return spec

    def get_params(self, i):
        return dict(self.params.iloc[i])

    def build(self):
        super(SurveyDatasetBuilder, self).build()

        # TODO: only copy wavelength if constant wave is use, otherwise the dataset
        #       builder should fill in the wave array one spectrum at a time
        raise NotImplementedError()

        self.dataset.wave[:] = self.pipeline.get_wave()
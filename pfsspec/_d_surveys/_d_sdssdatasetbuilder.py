from pfsspec.surveys.surveydatasetbuilder import SurveyDatasetBuilder

class SdssDatasetBuilder(SurveyDatasetBuilder):
    def __init__(self, orig=None, random_seed=None):
        super(SdssDatasetBuilder, self).__init__(orig=orig, random_seed=random_seed)
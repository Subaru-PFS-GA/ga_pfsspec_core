from pfsspec.surveys.surveyspectrum import SurveySpectrum

class SdssSpectrum(SurveySpectrum):
    """
    Mixin that implements additional parameters for SDSS survey data.
    """

    def __init__(self, orig=None):
        super(SdssSpectrum, self).__init__(orig=orig)

        if isinstance(orig, SdssSpectrum):
            self.mjd = orig.mjd
            self.plate = orig.plate
            self.fiber = orig.fiber
        else:
            self.mjd = None
            self.plate = None
            self.fiber = None

    def get_param_names(self):
        params = ['mjd',
                  'plate',
                  'fiber']
        return params

    def print_info(self):
        print('mjd=', self.mjd)
        print('plate=', self.plate)
        print('fiber=', self.fiber)
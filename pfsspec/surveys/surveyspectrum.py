from pfsspec.common.spectrum import Spectrum

class SurveySpectrum():
    """
    Mixin that implements parameters for survey data.
    """

    def __init__(self, orig=None):
        if isinstance(orig, SurveySpectrum):
            self.ra = orig.ra
            self.dec = orig.dec
        else:
            self.ra = None
            self.dec = None

    def get_param_names(self):
        params = ['ra',
                  'dec']
        return params

    def print_info(self):
        print('ra=', self.ra)
        print('dec=', self.dec)
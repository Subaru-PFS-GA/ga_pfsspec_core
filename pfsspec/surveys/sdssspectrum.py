from pfsspec.obsmod.spectrum import Spectrum

class SdssSpectrum():
    def __init__(self, orig=None):
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
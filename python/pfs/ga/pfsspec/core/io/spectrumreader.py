from pfs.ga.pfsspec.core.pfsobject import PfsObject

class SpectrumReader(PfsObject):
    """
    Implements functions to read one a more files from a spectrum file, stored
    in a format depending on the derived classes' implementation.
    """

    def __init__(self, wave_lim=None, orig=None):
        
        if not isinstance(orig, SpectrumReader):
            self.wave_lim = wave_lim
        else:
            self.wave_lim = wave_lim or orig.wave_lim

    def add_args(self, parser):
        # TODO: This collides with --lambda from ModelGrid
        #       Make sure a wavelength limit parameter is registered when
        #       building datasets
        parser.add_argument("--lambda", type=float, nargs=2, default=None, help="Wavelength limits.\n")

    def init_from_args(self, args):
        self.wave_lim = self.get_arg('lambda', self.wave_lim, args)

    def read(self):
        raise NotImplementedError()

    def read_all(self):
        return [self.read(),]
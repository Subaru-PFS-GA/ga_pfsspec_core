from pfsspec.core.pfsobject import PfsObject

class SpectrumReader(PfsObject):
    """
    Implements functions to read one a more files from a spectrum file, stored
    in a format depending on the derived classes' implementation.
    """

    def __init__(self, orig=None):
        pass

    def add_args(self, parser):
        pass

    def init_from_args(self, args):
        pass

    def read(self):
        raise NotImplementedError()

    def read_all(self):
        return [self.read(),]
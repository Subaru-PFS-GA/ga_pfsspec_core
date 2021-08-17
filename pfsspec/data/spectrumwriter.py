from pfsspec.common.pfsobject import PfsObject

class SpectrumWriter(PfsObject):
    def __init__(self, orig=None):
        super(SpectrumWriter, self).__init__(orig=orig)

    def write(self, file, spec):
        raise NotImplementedError()
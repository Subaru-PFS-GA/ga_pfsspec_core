from ...pfsobject import PfsObject

class FluxCorrectionModel(PfsObject):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

    def get_param_count(self):
        raise NotImplementedError()
from pfs.ga.pfsspec.core import PfsObject

# TODO: Inherit from Plugin
class Exporter(PfsObject):
    def __init__(self, orig=None):
        super(Exporter, self).__init__()

    def add_args(self, parser, config):
        pass

    def init_from_args(self, config, args):
        pass

    def execute_notebooks(self, script):
        pass
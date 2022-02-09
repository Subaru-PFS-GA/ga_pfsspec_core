from pfsspec.core import PfsObject

class Importer(PfsObject):
    def __init__(self, orig=None):
        super(Importer, self).__init__()

    def add_subparsers(self, configurations, parser):
        return []

    def add_args(self, parser):
        pass

    def init_from_args(self, config, args):
        pass

    def execute_notebooks(self, script):
        pass
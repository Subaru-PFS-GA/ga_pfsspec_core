from pfsspec.core import PfsObject

class Downloader(PfsObject):
    def __init__(self, orig=None):
        super(Downloader, self).__init__()

        if not isinstance(orig, Downloader):
            self.resume = False
            self.outdir = None
        else:
            self.resume = orig.resume
            self.outdir = orig.outdir

    def add_subparsers(self, configurations, parser):
        return []

    def add_args(self, parser):
        pass

    def init_from_args(self, config, args):
        pass

    def execute_notebooks(self, script):
        pass
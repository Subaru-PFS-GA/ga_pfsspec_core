from pfs.ga.pfsspec.core.scripts import Plugin

class Downloader(Plugin):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, Downloader):
            self.outdir = None
        else:
            self.outdir = orig.outdir

    def add_subparsers(self, configurations, parser):
        return []

    def add_args(self, parser, config):
        super().add_args(parser, config)

    def init_from_args(self, config, args):
        super().init_from_args(config, args)

    def execute_notebooks(self, script):
        pass
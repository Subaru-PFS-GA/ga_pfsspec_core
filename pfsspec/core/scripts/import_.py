import os
import logging

from .script import Script

class Import(Script):

    CONFIG_NAME = 'IMPORT_CONFIGURATIONS'

    def __init__(self):
        super(Import, self).__init__()

        self.path = None
        self.outdir = None
        self.resume = False

        self.importer = None

    # TODO: move to script?
    def add_subparsers(self, parser):
        # Register two positional variables that determine the importer class
        # and subclass
        cps = parser.add_subparsers(dest=self.CONFIG_CLASS)
        for c in self.parser_configurations:
            cp = cps.add_parser(c)
            sps = cp.add_subparsers(dest=self.CONFIG_SUBCLASS)
            for s in self.parser_configurations[c]:
                sp = sps.add_parser(s)

                # Instantiate importer and register further subparsers
                importer = self.parser_configurations[c][s][self.CONFIG_TYPE]()
                subparsers = importer.add_subparsers(self.parser_configurations[c][s], sp)
                if subparsers is not None:
                    for ss in subparsers:
                        self.add_args(ss)
                else:
                    self.add_args(sp)

    def add_args(self, parser):
        super(Import, self).add_args(parser)
        parser.add_argument("--in", type=str, required=True, help="Model/data directory base path\n")
        parser.add_argument("--out", type=str, required=True, help="Output file, must be .h5 or .npz\n")
        parser.add_argument('--resume', action='store_true', help='Resume existing but aborted import.\n')

    def parse_args(self):
        super(Import, self).parse_args()

        self.resume = self.get_arg('resume', self.resume)

    def create_importer(self):
        config = self.parser_configurations[self.args[self.CONFIG_CLASS]][self.args[self.CONFIG_SUBCLASS]]
        self.importer = config[self.CONFIG_TYPE]()
        self.importer.parallel = self.threads != 1
        self.importer.threads = self.threads
        self.importer.resume = self.resume
        self.importer.init_from_args(config, self.args)

        # TODO: initialize pipeline

    def open_data(self):
        self.importer.open_data(self.args['in'], self.outdir)

    def prepare(self):
        super(Import, self).prepare()
        
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=self.resume)
        self.init_logging(self.outdir)

        self.create_importer()
        self.open_data()

    def run(self):
        self.importer.run()
        self.importer.save_data()

    def finish(self):
        self.importer.execute_notebooks(self)

def main():
    script = Import()
    script.execute()

if __name__ == "__main__":
    main()
import os
import logging

#from pfsspec.scripts.configurations import IMPORT_CONFIGURATIONS
from .script import Script

class Import(Script):

    CONFIG_NAME = 'IMPORT_CONFIGURATIONS'

    def __init__(self):
        super(Import, self).__init__()

        self.config_name = None

        self.path = None
        self.outdir = None
        self.resume = False

        self.importer = None

    def add_subparsers(self, parser):
        tps = parser.add_subparsers(dest='type')
        for t in self.parser_configurations:
            tp = tps.add_parser(t)
            sps = tp.add_subparsers(dest='source')
            for s in self.parser_configurations[t]:
                sp = sps.add_parser(s)
                self.add_args(sp)
                rr = self.parser_configurations[t][s]()
                rr.add_args(sp)

    def add_args(self, parser):
        super(Import, self).add_args(parser)
        parser.add_argument("--in", type=str, required=True, help="Model/data directory base path\n")
        parser.add_argument("--out", type=str, required=True, help="Output file, must be .h5 or .npz\n")
        parser.add_argument('--resume', action='store_true', help='Resume existing but aborted import.\n')

    def parse_args(self):
        super(Import, self).parse_args()

        self.resume = self.get_arg('resume', self.resume)

    def create_importer(self):
        self.importer = self.parser_configurations[self.args['type']][self.args['source']]()
        self.importer.parallel = self.threads != 1
        self.importer.threads = self.threads
        self.importer.resume = self.resume
        self.importer.init_from_args(self.args)

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
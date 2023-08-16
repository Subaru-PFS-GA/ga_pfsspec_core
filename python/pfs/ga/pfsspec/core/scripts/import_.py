#!/usr/bin/env python3

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

    def add_args(self, parser, config):
        super(Import, self).add_args(parser, config)

        parser.add_argument("--in", type=str, required=True, help="Model/data directory base path\n")
        parser.add_argument("--out", type=str, required=True, help="Output file, must be .h5 or .npz\n")

    def parse_args(self):
        super(Import, self).parse_args()

        self.resume = self.get_arg('resume', self.resume)

    def create_importer(self):
        config = self.parser_configurations[self.args[self.CONFIG_CLASS]][self.args[self.CONFIG_SUBCLASS]]
        self.importer = config[self.CONFIG_TYPE]()
        self.importer.init_from_args(self, config, self.args)

    def open_data(self, args):
        self.importer.open_data(args, args['in'], self.outdir)

    def prepare(self):
        super(Import, self).prepare()
        
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=self.resume)
        self.init_logging(self.outdir)

        self.create_importer()
        self.open_data(self.args)

    def run(self):
        self.importer.run()
        self.importer.save_data(self.args, self.outdir)

    def finish(self):
        self.importer.execute_notebooks(self)

def main():
    script = Import()
    script.execute()

if __name__ == "__main__":
    main()
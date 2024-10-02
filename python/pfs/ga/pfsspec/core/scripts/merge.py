#!/usr/bin/env python3

import numpy as np
import pandas as pd

from ..setup_logger import logger
from pfs.ga.pfsspec.core.scripts import Script

class Merge(Script):

    CONFIG_NAME = 'MERGE_CONFIGURATIONS'

    def __init__(self):
        super(Merge, self).__init__()

        self.merger = None

    def add_subparsers(self, parser):
        sps = parser.add_subparsers(dest='source', required=True)
        for src in self.parser_configurations:
            ps = sps.add_parser(src)
            spd = ps.add_subparsers(dest='type', required=True)
            for dt in self.parser_configurations[src]:
                mrconfig = self.parser_configurations[src][dt]
                
                merger = self.create_merger(mrconfig)
                
                pp = spd.add_parser(dt)

                # Register merger arguments
                merger.add_args(pp, mrconfig)

    def add_args(self, parser, config):
        super(Merge, self).add_args(parser, config)
        parser.add_argument("--in", type=str, nargs='+', help="List of inputs.\n")
        parser.add_argument("--out", type=str, help="Output directory\n")

    def parse_args(self):
        super(Merge, self).parse_args()

    def create_merger(self, mrconfig):
        merger = mrconfig['type']()
        return merger
    
    def load_input_data(self, config):
        self.merger.load_input(self.args['in'])

    def init_merger(self, config):
        self.merger.init_from_args(self, config, self.args)

    def init_input_data(self):
        pass

    def init_output_data(self):
        pass

    def prepare(self):
        super(Merge, self).prepare()
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=False)
        self.init_logging(self.outdir)

        mrconfig = self.parser_configurations[self.args['source']][self.args['type']]

        self.merger = self.create_merger(mrconfig)
        self.init_merger(mrconfig)

        self.init_input_data()
        self.load_input_data()

        self.init_output_data()

    def run(self):
        self.merger.merge()

        logger.info('Results are written to {}'.format(self.args['out']))

    def execute_notebooks(self):
        super(Merge, self).execute_notebooks()

        # self.execute_notebook('eval_dataset', parameters={
        #                           'DATASET_PATH': self.args['out']
        #                       })

def main():
    script = Merge()
    script.execute()

if __name__ == "__main__":
    main()

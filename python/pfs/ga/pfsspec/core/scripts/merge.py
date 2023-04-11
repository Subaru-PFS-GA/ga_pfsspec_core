#!/usr/bin/env python3

import os
import gc
import sys
import logging
import json
import numpy as np
import pandas as pd

from pfs.ga.pfsspec.core import Constants
from pfs.ga.pfsspec.core.dataset import Dataset
from pfs.ga.pfsspec.core.scripts import Script

from pfs.ga.pfsspec.data.datasetmerger import DatasetMerger

class Merge(Script):
    MERGER_TYPES = {
        'dataset': DatasetMerger
    }

    def __init__(self, logging_enabled=True):
        super(Merge, self).__init__(logging_enabled=logging_enabled)

        self.input_pattern = None
        self.merger = None

    def add_subparsers(self, parser):
        mps = parser.add_subparsers(dest='merger')
        for mm in Merge.MERGER_TYPES:
            mp = mps.add_parser(mm)
            self.add_args(mp)
            Merge.MERGER_TYPES[mm]().add_args(mp)

    def add_args(self, parser, config):
        super(Merge, self).add_args(parser, config)
        parser.add_argument("--in", type=str, nargs='+', help="List of inputs.\n")
        parser.add_argument("--out", type=str, help="Output directory\n")

        parser.add_argument('--skip-notebooks', action='store_true', help='Skip notebook step.\n')

    def parse_args(self):
        super(Merge, self).parse_args()

    def init_merger(self):
        # TODO: rewrite to use configurations
        self.merger = Merge.MERGER_TYPES[self.args['merger']]()
        self.merger.init_from_args(config, self.args)
        self.merger.init_inputs()
        self.merger.init_output()

    def prepare(self):
        super(Merge, self).prepare()
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=False)
        self.init_logging(self.outdir)

    def run(self):
        self.init_merger()
        self.merger.merge()

        self.logger.info('Results are written to {}'.format(self.args['out']))

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

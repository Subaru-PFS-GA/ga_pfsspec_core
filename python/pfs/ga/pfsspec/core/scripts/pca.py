#!/usr/bin/env python3

import os
import logging
import numpy as np

from .script import Script

class Pca(Script):

    CONFIG_NAME = 'PCA_CONFIGURATIONS'

    def __init__(self):
        super(Pca, self).__init__()

        self.pca = None

    def add_args(self, parser, config):
        super(Pca, self).add_args(parser, config)

        parser.add_argument('--in', type=str, help="Input data path.\n")
        parser.add_argument('--out', type=str, help='Output data path.\n')
        parser.add_argument('--params', type=str, help='Parameters grid, if different from input.\n')
        parser.add_argument('--weights', type=str, help='Weights grid, if different from input.\n')

    def create_plugin(self, config):
        t = config[self.CONFIG_TYPE]
        if t is not None:
            return t(config['config'])
        else:
            return None

    def create_pca(self):
        config = self.parser_configurations[self.args[self.CONFIG_CLASS]][self.args[self.CONFIG_SUBCLASS]]
        self.pca = self.create_plugin(config)
        self.pca.init_from_args(config, self.args)

    def open_data(self, args):
        self.pca.open_data(args, args['in'], self.outdir, args['params'], args['weights'])

    def save_data(self, args):
        self.pca.save_data(args, self.outdir)

    def prepare(self):
        super(Pca, self).prepare()

        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=False)
        self.init_logging(self.outdir)

        self.create_pca()
        self.open_data(self.args)

    def run(self):
        self.pca.run()
        self.save_data(self.args)

def main():
    script = Pca()
    script.execute()

if __name__ == "__main__":
    main()
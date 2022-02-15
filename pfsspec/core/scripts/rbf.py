import os
import logging
import numpy as np

from .script import Script

class Rbf(Script):

    CONFIG_NAME = 'RBF_CONFIGURATIONS'

    def __init__(self):
        super(Rbf, self).__init__()

        self.outdir = None

        self.rbf = None

    def add_args(self, parser):
        super(Rbf, self).add_args(parser)

        parser.add_argument('--in', type=str, help="Input data path.\n")
        parser.add_argument('--out', type=str, help='Output data path.\n')
        parser.add_argument('--params', type=str, help='Parameters grid, if different from input.\n')

    def create_plugin(self, config):
        return config[self.CONFIG_TYPE](config['config'])

    def create_rbf(self):
        config = self.parser_configurations[self.args[self.CONFIG_CLASS]][self.args[self.CONFIG_SUBCLASS]]
        self.rbf = self.create_plugin(config)
        self.rbf.init_from_args(config, self.args)

    def open_data(self):
        self.rbf.open_data(self.args['in'], self.outdir, self.args['params'])

    def save_data(self):
        self.rbf.save_data(self.outdir)

    def prepare(self):
        super(Rbf, self).prepare()

        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=False)
        self.init_logging(self.outdir)

        self.create_rbf()
        self.open_data()

    def run(self):
        
        self.rbf.run()
        self.save_data()

def main():
    script = Rbf()
    script.execute()

if __name__ == "__main__":
    main()
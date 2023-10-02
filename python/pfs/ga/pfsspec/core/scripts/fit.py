#!/usr/bin/env python3

import os
import logging

from .script import Script

class Fit(Script):

    CONFIG_NAME = 'FIT_CONFIGURATIONS'

    def __init__(self):
        super(Fit, self).__init__()

        self.outdir = None

        self.fit = None

    def add_args(self, parser, config):
        super(Fit, self).add_args(parser, config)

        parser.add_argument('--in', type=str, help="Input data path.\n")
        parser.add_argument('--out', type=str, help='Output data path.\n')
        parser.add_argument('--params', type=str, help='Parameters grid, if different from input.\n')

    def create_plugin(self, config):
        t = config[self.CONFIG_TYPE]
        if t is not None:
            return t(config['config'])
        else:
            return None

    def create_fit(self):
        config = self.parser_configurations[self.args[self.CONFIG_CLASS]][self.args[self.CONFIG_SUBCLASS]]
        self.fit = self.create_plugin(config)
        self.fit.parallel = self.threads != 1
        self.fit.threads = self.threads
        self.fit.init_from_args(config, self.args)

    def open_data(self, args):
        self.fit.open_data(args, args['in'], self.outdir, args['params'])

    def prepare(self):
        super(Fit, self).prepare()

        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=False)
        self.init_logging(self.outdir)

        self.create_fit()
        self.open_data(self.args)

    def run(self):
        self.fit.run()
        self.fit.save_data(self.args, self.outdir)

    def finish(self):
        self.fit.execute_notebooks(self)

def main():
    script = Fit()
    script.execute()

if __name__ == "__main__":
    main()
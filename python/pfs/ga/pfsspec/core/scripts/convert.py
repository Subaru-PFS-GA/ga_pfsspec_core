import os
import logging

from .script import Script

class Convert(Script):

    CONFIG_NAME = 'CONVERT_CONFIGURATIONS'

    def __init__(self):
        super().__init__()

        self.outdir = None
        self.resume = False

        self.converter = None

    def add_args(self, parser):
        super().add_args(parser)

        parser.add_argument("--in", type=str, required=True, help="Model/data directory base path\n")
        parser.add_argument("--out", type=str, required=True, help="Output file, must be .h5 or .npz\n")
        parser.add_argument('--resume', action='store_true', help='Resume existing but aborted conversion.\n')

    def parse_args(self):
        super().parse_args()

        self.resume = self.get_arg('resume', self.resume)

    def create_plugin(self, config):
        return config[self.CONFIG_TYPE](config['config'])

    def create_converter(self):
        config = self.parser_configurations[self.args[self.CONFIG_CLASS]][self.args[self.CONFIG_SUBCLASS]]
        self.converter = self.create_plugin(config)
        self.converter.parallel = self.threads != 1
        self.converter.threads = self.threads
        self.converter.init_from_args(config, self.args)

    def open_data(self, args):
        self.converter.open_data(args, args['in'], self.outdir)

    def prepare(self):
        super().prepare()
        
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=self.resume)
        self.init_logging(self.outdir)

        self.create_converter()
        self.open_data(self.args)

    def run(self):
        self.converter.run()
        self.converter.save_data(self.args, self.outdir)

    def finish(self):
        self.converter.execute_notebooks(self)

def main():
    script = Convert()
    script.execute()

if __name__ == "__main__":
    main()
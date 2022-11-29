#!/usr/bin/env python3

import os
import logging

from .script import Script

class Download(Script):

    CONFIG_NAME = 'DOWNLOAD_CONFIGURATIONS'

    def __init__(self):
        super(Download, self).__init__()

        self.outdir = None
        self.resume = False

        self.downloader = None

    def add_args(self, parser):
        super(Download, self).add_args(parser)

        parser.add_argument("--out", type=str, required=True, help="Output directory\n")
        parser.add_argument('--resume', action='store_true', help='Resume existing but aborted download.\n')

    def parse_args(self):
        super(Download, self).parse_args()

        self.resume = self.get_arg('resume', self.resume)

    def create_downloader(self):
        config = self.parser_configurations[self.args[self.CONFIG_CLASS]][self.args[self.CONFIG_SUBCLASS]]
        self.downloader = config[self.CONFIG_TYPE]()
        self.downloader.parallel = self.threads != 1
        self.downloader.threads = self.threads
        self.downloader.resume = self.resume
        self.downloader.outdir = self.outdir
        self.downloader.init_from_args(config, self.args)

    def prepare(self):
        super(Download, self).prepare()
        
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir, resume=self.resume)
        self.init_logging(self.outdir)

        self.create_downloader()

    def run(self):
        self.downloader.run()

    def finish(self):
        self.downloader.execute_notebooks(self)
        
def main():
    script = Download()
    script.execute()

if __name__ == "__main__":
    main()
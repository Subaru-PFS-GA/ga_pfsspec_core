#!/usr/bin/env python3

import os
import logging
import numpy as np

from pfs.ga.pfsspec.core.scripts import Script
from pfs.ga.pfsspec.data.dataset import Dataset
from pfs.ga.pfsspec.data.datasetspectrumwriter import DatasetSpectrumWriter

class Export(Script):
    def __init__(self):
        super(Export, self).__init__()

        self.outdir = None
        self.writer = None


    def add_args(self, parser, config):
        super(Export, self).add_args(parser, config)
        parser.add_argument('--in', type=str, help="Data set directory\n")
        parser.add_argument('--out', type=str, help='Export output directory\n')

    def parse_args(self):
        super(Export, self).parse_args()

    def create_writer(self):
        return DatasetSpectrumWriter()

    def create_dataset(self):
        return Dataset()

    def create_pipeline(self):
        return None

    def load_data(self):
        self.writer = self.create_writer()
        self.writer.outdir = self.outdir
        self.writer.pipeline = self.create_pipeline()
        self.writer.dataset = self.create_dataset()
        self.writer.dataset.load(os.path.join(self.args['in'], 'dataset.h5'), format('h5'))

    def prepare(self):
        super(Export, self).prepare()
        self.outdir = self.args['out']
        self.create_output_dir(self.outdir)
        self.init_logging(self.outdir)

        self.load_data()

    def run(self):
        self.writer.write_all()
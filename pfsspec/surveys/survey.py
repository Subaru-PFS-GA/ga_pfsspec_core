import logging
import numpy as np
import pandas as pd
import pickle

class Survey():
    def __init__(self):
        self.params = None
        self.spectra = None

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)
            pickle.dump(self.spectra, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
            self.spectra = pickle.load(f)

        self.logger.info("Loaded survey with shapes:")
        self.logger.info("  spectra: {}".format(len(self.spectra)))
        self.logger.info("  params:  {}".format(self.params.shape))
        self.logger.info("  columns: {}".format(self.params.columns))

    def add_args(self, parser):
        pass

    def init_from_args(self, args):
        pass
import logging
import numpy as np
import pandas as pd
import pickle

from pfsspec.common.pfsobject import PfsObject

class Survey(PfsObject):
    """
    Implements functions to store survey data of any type of Spectrum implementation.

    Stores data pickled.
    """

    # TODO: Can we easily rewrite this to store spectra in HDF5. Spectra are not
    #       supposed to be uniform here but we still can pickle one by one and
    #       store as variable length binaries or jagged arrays.

    def __init__(self, orig=None):
        super(Survey, self).__init__(orig=orig)

        if isinstance(orig, Survey):
            self.params = orig.params
            self.spectra = orig.spectra
        else:
            self.params = None
            self.spectra = None

    def save(self, filename=None, format=None):
        self.filename = filename or self.filename
        self.format = format or self.fileformat

        with open(self.filename, 'wb') as f:
            pickle.dump(self.params, f)
            pickle.dump(self.spectra, f)

    def load(self, filename=None):
        self.filename = filename or self.filename
        self.fileformat = format or self.fileformat

        with open(self.filename, 'rb') as f:
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
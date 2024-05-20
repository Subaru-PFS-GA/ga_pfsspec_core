import numpy as np
import scipy.stats
import pandas as pd

from ..setup_logger import logger
from .dataset import Dataset

class PcaDataset(Dataset):
    def __init__(self, orig=None, preload_arrays=False):
        super(PcaDataset, self).__init__(orig=orig, preload_arrays=preload_arrays)

        if isinstance(orig, PcaDataset):
            self.U = orig.U
            self.S = orig.S
            self.V = orig.V
            self.PC = orig.PC
        else:
            self.U = None
            self.S = None
            self.V = None
            self.PC = None

    def run_pca(self, truncate=None):
        if self.wave.ndim != 1:
            raise Exception('PCA is only meaningful with a constant wave grid.')

        C = np.dot(self.flux.transpose(), self.flux)
        self.U, self.S, self.V = np.linalg.svd(C)

        if truncate is not None:
            self.PC = np.dot(self.flux, self.U[:, 0:truncate])
        else:
            self.PC = np.dot(self.flux, self.U[:, :])

        self.wave = np.arange(self.PC.shape[1])
        self.flux = self.PC
        self.error = np.zeros(self.flux.shape)
        self.mask = np.full(self.flux.shape, False)

    def save_pca(self, filename, format=None):
        logger.info("Saving PCA eigensystem to file {}...".format(filename))
        self.save(filename, format=format, save_items_func=self.save_pca_items)
        logger.info("Saved PCA eigensystem.")

    def save_pca_items(self):
        self.save_item('U', self.U)
        self.save_item('S', self.S)
        self.save_item('V', self.V)

    def load_pca(self, filename, s=None, format=None):
        logger.info("Loading PCA eigensystem from file {}...".format(filename))
        self.load(filename, s=s, format=format, load_items_func=self.load_pca_items)
        logger.info("Loaded PCA eigensystem.")

    def load_pca_items(self, s=None):
        self.U = self.load_item('U', np.ndarray)
        self.S = self.load_item('S', np.ndarray)
        self.V = self.load_item('V', np.ndarray)

    # TODO: split, filter, merge, etc.
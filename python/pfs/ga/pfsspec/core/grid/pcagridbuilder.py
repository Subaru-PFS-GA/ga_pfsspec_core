import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
import logging
from tqdm import tqdm

from pfs.ga.pfsspec.core.util.timer import Timer
from .pcagrid import PcaGrid
from .gridbuilder import GridBuilder

class PcaGridBuilder(GridBuilder):
    def __init__(self, input_grid=None, output_grid=None, orig=None):
        super(PcaGridBuilder, self).__init__(input_grid=input_grid, output_grid=output_grid, orig=orig)

        if not isinstance(orig, PcaGridBuilder):
            self.pca_method = 'cov'
            self.pca_subtract_mean = False
            self.pca_use_weights = True
            self.pca_transform = 'none'
            self.pca_norm = 'none'
            self.pca_truncate = None
            self.svd_method = 'svd'

            self.X = None           # Data matrix
            self.R = None           # Residual after truncation
            self.W = None           # Weight vector
            self.M = None           # Mean vector
            self.S = None           # Singular values (square root of covariance matrix eigenvalues)
            self.V = None           # Singular vectors (covariance matrix eigenvector)
            self.PC = None          # Principal components
        else:
            self.pca_method = orig.pca_method
            self.pca_subtract_mean = orig.pca_subtract_mean
            self.pca_use_weights = orig.pca_use_weights
            self.pca_transform = orig.pca_transform
            self.pca_norm = orig.pca_norm
            self.pca_truncate = orig.pca_truncate
            self.svd_method = orig.svd_method

            self.X = orig.X
            self.R = orig.R
            self.W = orig.W
            self.M = orig.M
            self.S = orig.S
            self.V = orig.V
            self.PC = orig.PC

    def add_args(self, parser, config):
        super(PcaGridBuilder, self).add_args(parser, config)

        parser.add_argument('--pca-method', type=str, default='cov', choices=['svd', 'cov', 'dual'], help='PCA method\n')
        parser.add_argument('--pca-subtract-mean', action='store_true', help='Subtract mean from data matrix\n')
        parser.add_argument('--pca-use-weights', action='store_true', help='Use vector weights (if available).\n')
        parser.add_argument('--pca-transform', type=str, default='none', choices=['none'] + list(PcaGrid.PCA_TRANSFORM_FUNCTIONS.keys()), help='Transformation before PCA.')
        parser.add_argument('--pca-norm', type=str, default='none', choices=['none', 'sum'], help='Method of normalization before PCA.')
        parser.add_argument('--pca-truncate', type=int, default=None, help='Truncate SVD')
        parser.add_argument('--svd-method', type=str, default='svd', choices=['svd', 'trsvd', 'skip'], help='Truncate PCA')

    def init_from_args(self, config, args):
        super(PcaGridBuilder, self).init_from_args(config, args)

        self.pca_method = self.get_arg('pca_method', self.pca_method, args)
        self.pca_subtract_mean = self.get_arg('pca_subtract_mean', self.pca_subtract_mean, args)
        self.pca_use_weights = self.get_arg('pca_use_weights', self.pca_use_weights, args)
        self.pca_transform = self.get_arg('pca_transform', self.pca_transform, args)
        self.pca_norm = self.get_arg('pca_norm', self.pca_norm, args)
        self.svd_method = self.get_arg('svd_method', self.svd_method, args)
        
        # Rename to pca-truncate
        self.pca_truncate = self.get_arg('pca_truncate', self.pca_truncate, args)
        if self.pca_truncate == 0:
            self.pca_truncate = None

    def get_vector_shape(self):
        # Return the shape of data vectors, a one element tuple
        raise NotImplementedError()

    def get_vector(self, i):
        # Return the ith data vector
        raise NotImplementedError()

    def run(self):
        self.X, self.W = self.get_data_matrix()

        # Transform
        if self.pca_transform is not None and self.pca_transform != 'none':
            self.X = PcaGrid.PCA_TRANSFORM_FUNCTIONS[self.pca_transform]['forward'](self.X)

        # Normalize the data vectors
        if self.pca_norm == 'none':
            pass
        elif self.pca_norm == 'sum':
            self.X = self.X / self.X.sum(axis=1)[..., np.newaxis]
        elif self.pca_norm is not None:
            raise NotImplementedError()
        
        # Subtract weighted mean
        if self.pca_subtract_mean:
            self.M = np.sum(self.W[:, np.newaxis] * self.X, axis=0) / np.sum(self.W)
            self.X = self.X - self.M
        else:
            self.M = np.zeros(self.X.shape[1], dtype=self.X.dtype)

        if self.pca_method == 'svd':
            self.run_pca_svd()
        elif self.pca_method == 'cov':
            self.run_pca_cov()
        elif self.pca_method == 'dual':
            self.run_dual_pca()
        else:
            raise NotImplementedError()

        self.truncate_basis()
        self.calculate_pc()
        self.calculate_residual()

    def get_data_matrix(self, step='pca'):
        # Copy all data vectors into a single matrix
        vector_count = self.get_input_count()
        vector_shape = self.get_vector_shape()
        data_shape = (vector_count,) + vector_shape

        # Build the data matrix
        X = np.empty(data_shape)
        self.logger.info('Allocated {} bytes for data matrix of shape {}.'.format(X.size * X.itemsize, X.shape))

        W = np.ones((vector_count,))
        self.logger.info('Allocated {} bytes for weight vector of shape {}.'.format(W.size * W.itemsize, W.shape))

        with Timer('Assembling data matrix...', logging.INFO):
            for i in tqdm(range(vector_count)):
                v, w = self.get_vector(i)
                
                X[i, :] = v
                W[i] = w

        # Verify data matrix
        if np.isnan(X).sum() > 0:
            raise Exception("Nan found in data matrix.")

        if np.isnan(W).sum() > 0:
            raise Exception("Nan found in weight matrix.")

        return X, W

    def run_pca_svd(self):
        with Timer('Computing SVD with method `{}`, truncated at {}...'.format(self.svd_method, self.pca_truncate), logging.INFO):
            mask = (self.W > 0)
            WX = 1 / np.sqrt(np.sum(self.W[mask])) * np.sqrt(self.W[mask][:, np.newaxis]) * self.X[mask]

            if self.svd_method == 'skip':
                logging.warning('Skipping SVD computation.')
                M, N = self.X[mask].shape
                K = min(M, N)
                self.S = np.zeros((K,))
                self.V = np.zeros((N, K))
            elif self.svd_method == 'svd' or self.pca_truncate is None:
                _, self.S, Vh = np.linalg.svd(WX, full_matrices=False)
                self.V = Vh.T
            elif self.svd_method == 'trsvd':
                svd = TruncatedSVD(n_components=self.pca_truncate)
                svd.fit(WX)                             # shape: (items, dim)
                self.S = svd.singular_values_           # shape: (truncate,)
                self.V = svd.components_.T              # shape: (dim, truncate)
            else:
                raise NotImplementedError()

    def run_pca_cov(self):

        # TODO: use weights
        raise NotImplementedError()

        with Timer('Calculating covariance matrix...', logging.INFO):
            C = 1 / np.sum(self.W) *  np.matmul(self.X.T, np.diag(self.W), self.X)        # shape: (dim, dim)
        self.logger.info('Allocated {} bytes for covariance matrix.'.format(C.size * C.itemsize))

        # Compute the SVD of the covariance matrix
        with Timer('Computing SVD with method `{}`, truncated at {}...'.format(self.svd_method, self.pca_truncate), logging.INFO):
            if self.svd_method == 'svd' or self.pca_truncate is None:
                _, self.S, Vh = np.linalg.svd(C, full_matrices=False)
                self.V = Vh.T
            elif self.svd_method == 'trsvd':
                svd = TruncatedSVD(n_components=self.pca_truncate)
                svd.fit(C)
                self.S = svd.singular_values_            # shape: (truncate,)
                self.V = svd.components_.transpose()     # shape: (dim, truncate)
            elif self.svd_method == 'skip':
                logging.warning('Skipping SVD computation.')
                self.S = np.zeros(C.shape[0])
                self.V = np.zeros(C.shape)
            else:
                raise NotImplementedError()

    def run_dual_pca(self):

        # TODO: use weights
        raise NotImplementedError()

        with Timer('Calculating X X^T matrix...', logging.INFO):
            self.D = np.matmul(self.X, self.X.T)        # shape: (items, items)
        self.logger.info('Allocated {} bytes for X X^T matrix.'.format(self.D.size * self.D.itemsize))

        with Timer('Computing SVD with method `{}`, truncated at {}...'.format(self.svd_method, self.pca_truncate), logging.INFO):
            if self.svd_method == 'svd' or self.pca_truncate is None:
                raise NotImplementedError()
            elif self.svd_method == 'trsvd':
                raise NotImplementedError()

    def truncate_basis(self):
        # Truncated basis
        if self.pca_truncate is None:
            pass
        else:
            self.S = self.S[:self.pca_truncate]
            self.V = self.V[:, :self.pca_truncate]

    def calculate_pc(self):
        # Calculate principal components and truncated basis
        with Timer('Calculating principal components...', logging.INFO):
            self.PC = np.dot(self.X, self.V)

    def calculate_residual(self):
        # Calculate the residual and from the truncated basis
        # shape: (count, wave)
        self.R = self.X - np.matmul(self.PC, self.V.T)

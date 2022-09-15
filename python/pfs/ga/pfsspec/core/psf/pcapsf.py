import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import TruncatedSVD

from .psf import Psf

class PcaPsf(Psf):
    """
    Computes the convolution of a data vector with a kernel expressed as
    a linear combination of basis function (derived from PCA) and the expansion
    coefficients are tabulated and interpolated as a function of wavelength.

    This kind of PSF representation uses a fixed kernel size with a fixed
    wavelength grid and fixed kernel wavelength steps.
    """

    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, PcaPsf):
            self.wave = None            # wavelength grid
            self.wave_edges = None
            self.mean = None            # mean kernel
            self.eigs = None            # eigenvalues
            self.eigv = None            # eigenvectors
            self.pc = None              # coeffs as functions of wavelength
        else:
            self.wave = orig.wave
            self.wave_edges = orig.wave_edges
            self.mean = orig.mean
            self.eigs = orig.eigs
            self.eigv = orig.eigv
            self.pc = orig.pc

    def save_items(self):
        self.save_item('wave', self.wave)
        self.save_item('wave_edges', self.wave_edges)
        self.save_item('dwave', self.dwave)
        self.save_item('mean', self.mean)
        self.save_item('eigs', self.eigs)
        self.save_item('eigv', self.eigv)
        self.save_item('pc', self.pc)

    def load_items(self, s=None):
        self.wave = self.load_item('wave', np.ndarray, None)
        self.wave_edges = self.load_item('wave_edges', np.ndarray, None)
        self.dwave = self.load_item('dwave', np.ndarray, None)
        self.mean = self.load_item('mean', np.ndarray, None)
        self.eigs = self.load_item('eigs', np.ndarray, None)
        self.eigv = self.load_item('eigv', np.ndarray, None)
        self.pc = self.load_item('pc', np.ndarray, None)

    @staticmethod
    def from_psf(source_psf, wave, size, normalize=True, truncate=None):
        # Precomputes the kernel as a function of wavelength on the provided grid
        
        s = np.s_[:truncate] if truncate is not None else np.s_[:]

        k, idx, shift = source_psf.eval_kernel(wave, size, normalize=normalize)
        
        M = np.mean(k, axis=0)

        if truncate is not None and truncate < size // 2:
            svd = TruncatedSVD(n_components=truncate * 2)
            svd.fit(k - M)
            Vt = svd.components_
            S = svd.singular_values_
        else:
            U, S, Vt = np.linalg.svd(k - M)
            
        PC = np.matmul(k, Vt[s].T)

        pca_psf = PcaPsf()
        pca_psf.wave = wave[-shift:shift]
        pca_psf.mean = M
        pca_psf.eigs = S[s]
        pca_psf.eigv = Vt[s]
        pca_psf.pc = PC
        
        return pca_psf

    def eval_kernel_impl(self, wave, size=None, normalize=True):
        """
        Return the matrix necessary to calculate the convolution with a varying kernel 
        using the direct matrix product method. This function is for compatibility with the
        base class and not used by the `convolve` implementation itself.
        """

        if size is not None:
            logging.warning('PCA PSF does not support overriding kernel size.')

        shift = -(self.eigv.shape[-1] // 2)
        idx = (np.arange(size) + shift) + np.arange(-shift, wave.size + shift)[:, np.newaxis]
        w = wave[-shift:+shift]
        pc = self.pc[-shift:+shift]

        k = self.mean + np.matmul(pc, self.eigv)
        
        if normalize:
            k /= np.sum(k, axis=-1, keepdims=True)

        return k, idx, shift

    def convolve(self, wave, values, errors=None, size=None, normalize=None):
        if size is not None:
            logging.warning('PCA PSF does not support overriding kernel size.')

        if normalize is not None:
            logging.warning('PCA PSF does not support renormalizing the precomputed kernel.')

        # Convolve value vector with each eigenfunction, than take linear combination with
        # the wave-dependent principal components

        if isinstance(values, np.ndarray):
            vv = [ values ]
        else:
            vv = values
        
        if isinstance(errors, np.ndarray):
            ee = [ errors ]
        else:
            ee = errors

        shift = -(self.eigv.shape[1] // 2)

        rv = []
        for v in vv:
            m = np.convolve(v, self.mean, mode='valid')
            c = np.empty((m.shape[0], self.eigv.shape[0]))
            for i in range(self.eigv.shape[0]):
                c[..., i] = np.convolve(v, self.eigv[i], mode='valid')
            rv.append(m + np.sum(self.pc * c, axis=-1))

        if ee is not None:
            re = []
            for e in ee:
                m = np.convolve(e**2, self.mean**2, mode='valid')
                c = np.empty((m.shape[0], self.eigv.shape[0]))
                for i in range(self.eigv.shape[0]):
                    c[..., i] = np.convolve(e**2, self.eigv[i]**2, mode='valid')
                re.append(np.sqrt(m + np.sum(self.pc**2 * c, axis=-1)))
        else:
            re = None

        if isinstance(values, np.ndarray):
            rv = rv[0]

        if isinstance(errors, np.ndarray):
            re = re[0]

        w = wave[-shift:+shift]

        return w, rv, re, shift
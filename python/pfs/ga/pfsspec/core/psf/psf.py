from collections import Iterable
import numpy as np

from ..pfsobject import PfsObject

class Psf(PfsObject):
    """
    When implemented in derived classes, it calculates the points spread
    function kernel and computes the convolution.
    """

    def __init__(self, reuse_kernel=False, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, Psf):
            self.reuse_kernel = reuse_kernel

            self.cached_kernel = None
            self.cached_shift = None
        else:
            self.reuse_kernel = reuse_kernel if reuse_kernel is not None else orig.reuse_kernel

            self.cached_kernel = orig.cached_kernel
            self.cached_shift = orig.cached_shift

    def get_size(self):
        if self.cached_kernel is not None:
            return self.cached_kernel.shape[-1]
        else:
            return None

    def get_shift(self, size=None):
        if self.cached_shift is not None:
            return self.cached_shift
        elif size is not None:
            return size // 2
        else:
            return None

    def get_kernel_at(self, lam, dwave, normalize=False):
        raise NotImplementedError()

    def get_kernel_impl(self, wave, size=None, normalize=None):
        """
        Calculate the kernel at each value of the wave vector, using the wave grid.
        Assume an odd value for size.
        """

        size = size if size is not None else 5
        normalize = normalize if normalize is not None else False

        shift = -(size // 2)
        idx = (np.arange(size) + shift) + np.arange(-shift, wave.size + shift)[:, np.newaxis]
        w = wave[-shift:+shift, np.newaxis]
        dw = wave[idx] - w
        k = self.get_kernel_at(w, dw, normalize=normalize)

        return k, shift

    def get_kernel(self, wave, size=None, normalize=None):

        if self.reuse_kernel and self.cached_kernel is not None:
            return self.cached_kernel, self.cached_shift

        k, shift = self.get_kernel_impl(wave, size=size, normalize=normalize)

        if self.reuse_kernel:
            self.cached_kernel = k
            self.cached_shift = shift
                
        return k, shift

    def normalize(self, k):
        return k / np.sum(k, axis=-1, keepdims=True)

    def convolve(self, wave, values, errors=None, size=None, normalize=None):
        """
        Convolve the vectors of the `values` list with a kernel returned by `get_kernel`.
        Works as numpy convolve with the option `valid`, i.e. the results will be shorter.
        """
        
        if isinstance(values, np.ndarray):
            vv = [ values ]
        else:
            vv = values
        
        if isinstance(errors, np.ndarray):
            ee = [ errors ]
        else:
            ee = errors
        
        k, shift = self.get_kernel(wave, size=size, normalize=normalize)

        rv = []
        for v in vv:
            rv.append(np.sum(v[-shift:+shift, np.newaxis] * k, axis=-1))

        # Also convolve the error vector assuming it contains uncorrelated sigmas
        if ee is not None:
            re = []
            for e in ee:
                re.append(np.sqrt(np.sum((e[-shift:+shift, np.newaxis] * k)**2, axis=-1)))
        else:
            re = None

        if isinstance(values, np.ndarray):
            rv = rv[0]

        if isinstance(errors, np.ndarray):
            re = re[0]

        return rv, re, shift
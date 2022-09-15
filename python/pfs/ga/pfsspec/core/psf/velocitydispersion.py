import numpy as np

from ..physics import Physics
from ..util.math import gauss
from .lambdapsf import LambdaPsf

class VelocityDispersion(LambdaPsf):
    def __init__(self, vdisp, reuse_kernel=False, orig=None):
        
        if not isinstance(orig, VelocityDispersion):
            self.vdisp = vdisp
        else:
            self.vdisp = vdisp or orig.vdisp
        
        self.z = Physics.vdisp_to_z(vdisp)

        def kernel(w, dw):
            s = self.z * w
            return gauss(dw, m=0, s=s)

        super().__init__(func=kernel, reuse_kernel=reuse_kernel, orig=orig)

    def get_optimal_size(self, wave, tol=1e-5):
        """
        Given a tolerance and a wave grid, return the optimal kernel size.
        """

        # Evaluate the kernel at the most extreme values of wave and find
        # where the kernel goes below `tol`.

        k = self.eval_kernel_at(wave[0], wave - wave[0], normalize=True)
        s1 = np.max(np.where(k > tol)[0]) * 2 + 1

        k = self.eval_kernel_at(wave[-1], wave[-1] - wave, normalize=True)
        s2 = np.max(wave.size - np.where(k > tol)[0]) * 2 + 1

        return max(s1, s2)
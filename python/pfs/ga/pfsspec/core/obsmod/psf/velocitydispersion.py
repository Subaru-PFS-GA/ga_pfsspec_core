import numpy as np

from ...physics import Physics
from ...util.math import gauss
from .lambdapsf import LambdaPsf

class VelocityDispersion(LambdaPsf):
    def __init__(self, vdisp, reuse_kernel=False, orig=None):
        
        if not isinstance(orig, VelocityDispersion):
            self.vdisp = vdisp
        else:
            self.vdisp = vdisp or orig.vdisp
        
        self.z = Physics.vel_to_z(vdisp)

        def kernel(w, dw):
            s = self.z * w
            return gauss(dw, m=0, s=s)

        super().__init__(func=kernel, reuse_kernel=reuse_kernel, orig=orig)
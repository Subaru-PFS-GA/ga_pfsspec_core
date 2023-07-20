import numpy as np
from scipy.stats import multivariate_normal

from .scipydistribution import ScipyDistribution

class MultivariateNormalDistribution(ScipyDistribution):
    # n * (n + 1) = k
    # n**2 + n - k = 0
    # (-b +/- sqrt(b**2 - 4 * a * c)) / 2 / a
    # (-1 +/- sqrt(1 + 4 * k)) / 2


    def init_from_args(self, args):
        n = (np.sqrt(1 + 4 * len(args)) - 1) / 2

        if np.floor(n) != n:
            raise Exception("Invalid number of arguments for multivariate normal distribution.")
        else:
            n = int(n)
            self.loc = np.array([float(a) for a in args[:n]])
            self.scale = np.array([float(a) for a in args[n:]]).reshape(n, n)

    def sample_impl(self, loc=None, scale=None, min=None, max=None, size=None, random_state=None):
        return multivariate_normal.rvs(mean=loc, cov=scale, size=size, random_state=random_state)
        
    def pdf_impl(self, x, loc=None, scale=None, min=None, max=None):
        return multivariate_normal.pdf(x, mean=loc, cov=scale)
        
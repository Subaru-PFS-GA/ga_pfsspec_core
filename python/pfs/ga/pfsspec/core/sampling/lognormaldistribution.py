from scipy.stats import lognorm

from .scipydistribution import ScipyDistribution

class LogNormalDistribution(ScipyDistribution):
    def __init__(self, shape=None, loc=None, scale=None, min=None, max=None, random_state=None, orig=None):
        super().__init__(min=min, max=max, random_state=random_state, orig=orig)

        if not isinstance(orig, LogNormalDistribution):
            self.shape = shape
        else:
            self.shape = shape if shape is not None else orig.shape

    def init_from_args(self, args):
        if len(args) == 0:
            pass
        elif len(args) == 1:
            [self.shape] = [float(a) for a in args]
        elif len(args) == 3:
            [self.shape, self.loc, self.scale] = [float(a) for a in args]
        elif len(args) == 5:
            [self.shape, self.loc, self.scale, self.min, self.max] = [float(a) for a in args]
        else:
            raise Exception("Invalid number of arguments for log-normal distribution.")

    def sample_impl(self, shape=None, loc=None, scale=None, min=None, max=None, size=None, random_state=None):
        kw = {}
        
        if loc is not None:
            kw['loc'] = loc

        if scale is not None:
            kw['scale'] = scale

        shape = shape if shape is not None else self.shape

        if min is None or max is None:
            return lognorm.rvs(shape, size=size, random_state=random_state, **kw)
        else:
            a, b = lognorm.cdf(shape, min, **kw), lognorm.cdf(shape, max, **kw)
            return lognorm.ppf(shape, random_state.uniform(a, b, size=size), **kw)
    
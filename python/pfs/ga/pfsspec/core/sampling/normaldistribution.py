from scipy.stats import norm

from .scipydistribution import ScipyDistribution

class NormalDistribution(ScipyDistribution):
    def init_from_args(self, args):
        if len(args) == 0:
            pass
        elif len(args) == 2:
            [self.loc, self.scale] = [float(a) for a in args]
        elif len(args) == 4:
            [self.loc, self.scale, self.min, self.max] = [float(a) for a in args]
        else:
            raise Exception("Invalid number of arguments for normal distribution.")

    def sample_impl(self, loc=None, scale=None, min=None, max=None, size=None, random_state=None):
        if min is None and max is None:
            return norm.rvs(loc=loc, scale=scale, size=size, random_state=random_state)
        elif min is None:
            raise NotImplementedError()
        elif max is None:
            raise NotImplementedError()
        else:
            a, b = norm.cdf([min, max], loc=loc, scale=scale)
            return norm.ppf(random_state.uniform(a, b, size=size), loc=loc, scale=scale)

    def pdf_impl(self, x, loc=None, scale=None, min=None, max=None):
        if min is None and max is None:
            return norm.pdf(x, loc=loc, scale=scale)
        elif min is None:
            raise NotImplementedError()
        elif max is None:
            raise NotImplementedError()
        else:
            a, b = norm.cdf([min, max], loc=loc, scale=scale)
            return norm.pdf(x, loc=loc, scale=scale) / (b - a)
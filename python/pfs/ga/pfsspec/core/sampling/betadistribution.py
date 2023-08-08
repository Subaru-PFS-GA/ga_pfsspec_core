from scipy.stats import beta

from .scipydistribution import ScipyDistribution

class BetaDistribution(ScipyDistribution):
    def __init__(self, a=None, b=None, loc=None, scale=None, min=None, max=None, random_state=None, orig=None):
        super().__init__(min=min, max=max, random_state=random_state, orig=orig)

        if not isinstance(orig, BetaDistribution):
            self.a = a
            self.b = b
        else:
            self.a = a if a is not None else orig.a
            self.b = b if b is not None else orig.b

    def init_from_args(self, args):
        if len(args) == 0:
            pass
        elif len(args) == 2:
            [self.a, self.b] = [float(v) for v in args]
        elif len(args) == 4:
            [self.a, self.b, self.loc, self.scale] = [float(v) for v in args]
        elif len(args) == 6:
            [self.a, self.b, self.loc, self.scale, self.min, self.max] = [float(v) for v in args]
        else:
            raise Exception("Invalid number of arguments for log-normal distribution.")

    def sample_impl(self, a=None, b=None, loc=None, scale=None, min=None, max=None, size=None, random_state=None):
        kw = {}
        
        if loc is not None:
            kw['loc'] = loc

        if scale is not None:
            kw['scale'] = scale

        a = a if a is not None else self.a
        b = b if b is not None else self.b

        if min is None and max is None:
            return beta.rvs(a, b, size=size, random_state=random_state, **kw)
        elif min is None:
            raise NotImplementedError()
        elif max is None:
            raise NotImplementedError()
        else:
            # TODO: maybe modify it to rescale results between min and max?

            a, b = beta.cdf(a, b, min, **kw), beta.cdf(a, b, max, **kw)
            return beta.ppf(a, b, random_state.uniform(a, b, size=size), **kw)
    
    def pdf_impl(self, x, a=None, b=None, loc=None, scale=None, min=None, max=None):
        kw = {}
        
        if loc is not None:
            kw['loc'] = loc

        if scale is not None:
            kw['scale'] = scale

        a = a if a is not None else self.a
        b = b if b is not None else self.b

        if min is None and max is None:
            return beta.pdf(x, a, b)
        elif min is None:
            raise NotImplementedError()
        elif max is None:
            raise NotImplementedError()
        else:
            # TODO: maybe modify it to rescale results between min and max?

            a, b = beta.cdf(a, b, min, **kw), beta.cdf(a, b, max, **kw)
            p = beta.pdf(x, a, b, **kw) / (b - a)
            return self.mask_pdf(x, p, min, max)

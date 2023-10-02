from .distribution import Distribution

class ScipyDistribution(Distribution):
    def __init__(self, loc=None, scale=None, min=None, max=None, random_state=None, orig=None):
        super().__init__(min=min, max=max, random_state=random_state, orig=orig)

        if not isinstance(orig, ScipyDistribution):
            self.loc = loc
            self.scale = scale
        else:
            self.loc = loc if loc is not None else orig.loc
            self.scale = scale if scale is not None else orig.scale

    def init_from_args(self, args):
        raise NotImplementedError()
    
    def sample(self, loc=None, scale=None, min=None, max=None, size=None, random_state=None, **kwargs):
        random_state = self.get_random_state(random_state)
        min, max = self.get_min_max(min, max)
        loc = loc if loc is not None else self.loc
        scale = scale if scale is not None else self.scale

        return self.sample_impl(loc=loc, scale=scale, min=min, max=max, size=size, random_state=random_state, **kwargs)

    def sample_impl(self, loc, scale, min, max, size, random_state, **kw):
        raise NotImplementedError()

    def pdf(self, x, loc=None, scale=None, min=None, max=None, **kwargs):
        min, max = self.get_min_max(min, max)
        loc = loc if loc is not None else self.loc
        scale = scale if scale is not None else self.scale

        return self.pdf_impl(x, loc=loc, scale=scale, min=min, max=max, **kwargs)
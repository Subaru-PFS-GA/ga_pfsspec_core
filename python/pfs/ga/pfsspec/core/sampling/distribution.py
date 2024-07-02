import numpy as np

class Distribution():
    def __init__(self, min=None, max=None, random_state=None, orig=None):
        if not isinstance(orig, Distribution):
            self.random_state = random_state

            self.min = min
            self.max = max
        else:
            self.random_state = random_state if random_state is not None else orig.random_state

            self.min = min if min is not None else orig.min
            self.max = max if max is not None else orig.max

    def copy(self):
        return type(self)(orig=self)

    @staticmethod
    def from_args(name, args, random_state=None):
        from . import DISTRIBUTIONS
        dist = DISTRIBUTIONS[name](random_state=random_state)
        if args is not None:
            dist.init_from_args(args)
        return dist

    def init_from_args(self, args):
        raise NotImplementedError()
    
    def has_min_max(self):
        return self.min is not None and self.max is not None
    
    def get_min_max(self, min, max):
        min = min if min is not None else self.min
        max = max if max is not None else self.max

        return min, max
    
    def get_random_state(self, random_state):
        if random_state is not None:
            return random_state
        elif self.random_state is not None:
            return self.random_state
        else:
            return np.random
    
    def sample(self, size=None, random_state=None):
        raise NotImplementedError()
    
    def mask_pdf(self, x, p, min, max):
        return np.where((min <= x) & (x < max), p, 0.0)
    
    def pdf(self, x):
        raise NotImplementedError()
    
    def log_pdf(self, x):
        p = self.pdf(x)
        with np.errstate(divide='ignore'):
            return np.where(p > 0, np.log(p), -np.inf)
    
    def generate_initial_value(self, step_size_factor=0.1):
        if self.has_min_max():
            x_0 = 0.5 * (self.min + self.max)
            bounds = [self.min, self.max]       # Both are set
            step = (self.max - self.min) * step_size_factor
        else:
            s = self.sample(size=10)
            x_0 = s.mean()
            step = s.std() * step_size_factor
            bounds = [self.min, self.max]       # Either can be None but not both
            
        return x_0, bounds, step
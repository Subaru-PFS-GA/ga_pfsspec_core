import numpy as np

from .distribution import Distribution

class UniformDistribution(Distribution):
    def __init__(self, min=None, max=None, random_state=None, orig=None):
        super().__init__(min=min, max=max, random_state=random_state, orig=orig)

    def init_from_args(self, args):
        if len(args) == 0:
            pass
        elif len(args) == 2:
            [self.min, self.max] = [float(a) for a in args]
        else:
            raise Exception("Invalid number of arguments for uniform distribution.")

    def sample(self, min=None, max=None, size=None, random_state=None):
        random_state = self.get_random_state(random_state)
        min, max = self.get_min_max(min, max)
        return random_state.uniform(min, max, size=size)
    
    def pdf(self, x, min=None, max=None):
        min, max = self.get_min_max(min, max)
        return np.broadcast_to(1.0 / (max - min), x.shape)
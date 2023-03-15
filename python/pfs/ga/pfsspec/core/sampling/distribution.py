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

    @staticmethod
    def from_args(args):
        raise NotImplementedError()

    def init_from_args(self, args):
        raise NotImplementedError()
    
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
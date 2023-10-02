import numpy as np

RANDOM_DISTS = ['uniform', 'normal', 'lognormal', 'beta']

def get_random_dist(dist, random_state):
    if dist is None or dist == 'uniform':
        return random_state.uniform
    elif dist == 'normal':
        return random_state.normal
    elif dist == 'lognormal':
        return random_state.lognormal
    elif dist == 'beta':
        def beta(a=None, b=None, size=None):
            if a is None or b is None:
                random_state.beta(0.7, 0.7, size=None)
            else:
                random_state.beta(a, b, size=None)
        return beta
    else:
        raise NotImplementedError()
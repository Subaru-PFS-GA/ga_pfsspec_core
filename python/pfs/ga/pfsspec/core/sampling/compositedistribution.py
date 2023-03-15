import numpy as np

from .distribution import Distribution

class CompositeDistribution(Distribution):
    def __init__(self, random_state=None, orig=None):
        super().__init__(random_state=random_state, orig=orig)

        if not isinstance(orig, CompositeDistribution):
            self.dists = None
            self.weights = None
        else:
            self.dists = orig.dists
            self.weights = orig.weights

    def init_from_args(self, args):
        # Arguments of the composite distribution:
        # - List of weights
        # - List of distributions in square brackets

        self.dists = []
        self.weights = []

        # Figure out the number of distributions, it comes from the number
        # of weights listed before the first opening square bracket [
        ndist = args.index('[')

        self.weights = np.array(args[:ndist], dtype=float)
        self.weights /= self.weights.sum()

        ss = ndist
        for i in range(ndist):
            se = args[ss:].index(']')
            aa = args[ss+1:ss+se]
            self.dists.append(Distribution.from_args(aa[0], aa[1:]))
            ss = ss + se + 1

    def sample(self, size=None, random_state=None):
        random_state = self.get_random_state(random_state)

        # Draw random numbers for each distribution in a vectorized way.
        # This is faster than drawing numbers one by one
        n = len(self.dists)
        t = random_state.choice(np.arange(n, dtype=int), p=self.weights, size=size)
        s = np.choose(t, [d.sample(size=size, random_state=random_state) for d in self.dists])

        return s
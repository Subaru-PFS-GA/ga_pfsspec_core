import numpy as np

from ..util.args import *
from .distribution import Distribution
from .uniformdistribution import UniformDistribution

class Parameter():
    def __init__(self, name, value=None, min=None, max=None, dist=None, dist_args=None, orig=None):
        if not isinstance(orig, Parameter):
            self.name = name
            self.value = value
            self.min = min
            self.max = max
            self.dist = dist
            self.dist_args = dist_args
        else:
            self.name = orig.name
            self.value = orig.value
            self.min = orig.min
            self.max = orig.max
            self.dist = orig.dist
            self.dist_args = orig.dist_args

    def add_args(self, parser):
        parser.add_argument(f'--{self.name}', type=float, nargs='*', default=self.value, help=f'Limit on {self.name}.\n')
        parser.add_argument(f'--{self.name}-dist', type=str, nargs='*', help=f'Distribution type and params for {self.name}.\n')

    def init_from_args(self, args):
        # Parse minimum, maximum and/or value for the parameter
        if is_arg(self.name, args):
            values = args[self.name]

            if len(values) >= 2:
                self.min = values[0]
                self.max = values[1]
            else:
                self.value = values[0]
                self.min = values[0]
                self.max = values[0]
            
        # Parse the distribution settings of the parameter. If no
        # distribution is specified but we have a min and max value
        # that are different, we sample from uniform by default.
        if self.min is not None and self.max is not None and self.min == self.max:
            self.dist = 'const'
            pass
        elif is_arg(f'{self.name}_dist', args):
            aa = get_arg(f'{self.name}_dist', self.dist, args)
            if not isinstance(aa, list):
                aa = [ aa ]

            self.dist = aa[0]
            self.dist_args = aa[1:]

    def get_dist(self, random_state=None):
        d = None

        if self.dist is None and self.min is not None and self.max is not None and self.min != self.max:
            d = UniformDistribution(self.min, self.max, random_state=random_state)
        else:
            d = Distribution.from_args(self.dist, self.dist_args, random_state=random_state)
            
            if self.min is not None:
                d.min = self.min

            if self.max is not None:
                d.max = self.max

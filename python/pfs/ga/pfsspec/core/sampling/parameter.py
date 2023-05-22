from collections.abc import Iterable
import numpy as np

from ..util.args import *
from .distribution import Distribution
from .uniformdistribution import UniformDistribution

class Parameter():
    def __init__(self, name, value=None, min=None, max=None, dist=None, dist_args=None, help=None, orig=None):
        if not isinstance(orig, Parameter):
            self.name = name
            self.value = value
            self.min = min
            self.max = max
            self.dist = dist
            self.dist_args = dist_args
            self.help = help
        else:
            self.name = orig.name
            self.value = orig.value
            self.min = orig.min
            self.max = orig.max
            self.dist = orig.dist
            self.dist_args = orig.dist_args
            self.help = orig.help

    def add_args(self, parser):
        parser.add_argument(f'--{self.name}', type=float, nargs='*', default=self.value, help=f'Limit on {self.name}.\n')
        parser.add_argument(f'--{self.name}-dist', type=str, nargs='*', help=f'Distribution type and params for {self.name}.\n')

    def init_from_args(self, args):
        # Parse minimum, maximum and/or value for the parameter
        if is_arg(self.name, args):
            values = args[self.name]
            if not isinstance(values, Iterable):
                values = [ values ]

            if len(values) == 1:
                self.value = values[0]
                self.min = values[0]
                self.max = values[0]
            elif len(values) == 2:
                self.min = values[0]
                self.max = values[1]
            else:
                raise ValueError(f'Invalid number of arguments for parameter `{self.name}`.')
            
        # Parse the distribution settings of the parameter. If no
        # distribution is specified but we have a min and max value
        # that are different, we sample from uniform by default.
        if self.min is not None and self.max is not None and self.min == self.max:
            self.dist = 'const'
        elif self.min is not None and self.max is not None and not is_arg(f'{self.name}_dist', args):
            self.dist = 'uniform'
            self.dist_args = []
        elif is_arg(f'{self.name}_dist', args):
            aa = get_arg(f'{self.name}_dist', self.dist, args)
            if not isinstance(aa, list):
                aa = [ aa ]

            self.dist = aa[0]
            self.dist_args = aa[1:]

    def has_value(self):
        return self.value is not None
    
    def has_min_max(self):
        return self.min is not None and self.max is not None

    def has_dist(self):
        return self.dist is not None and self.dist != 'const'
            
    def get_dist(self, random_state=None):
        d = None

        if self.dist is None and (self.min is None or self.max is None):
            d = None
        else:
            d = Distribution.from_args(self.dist, self.dist_args, random_state=random_state)
            
            if self.min is not None:
                d.min = self.min

            if self.max is not None:
                d.max = self.max

        return d
    
    def generate_initial_value(self, step_size_factor=0.1):
        """
        Generate a good initial value based on the limits and the specified
        distribution.
        """

        if self.has_value():
            # This is a fixed parameter
            x_0 = self.value
            bounds = [self.value, self.value]
            steps = np.nan
            fixed = True
        elif self.has_min_max():
            x_0 = 0.5 * (self.min + self.max)
            bounds = [self.min, self.max]
            steps = (self.max - self.min) * step_size_factor
            fixed = False
        elif self.has_dist():
            # This is a fitted (sampler) parameter. Sample initial value
            # from the distribution
            d = self.get_dist()
            x_0, bounds, steps = d.generate_initial_value(step_size_factor=step_size_factor)
            fixed = False
        else:
            raise Exception(f'Cannot generate initial value for parameter `{self.name}`.')
        
        return x_0, bounds, steps, fixed
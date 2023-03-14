import numpy as np

from ..util.args import *
from ..util.dist import RANDOM_DISTS

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
        if is_arg(self.name, args):
            values = args[self.name]

            if len(values) >= 2:
                self.min = values[0]
                self.max = values[1]
            else:
                self.value = values[0]
                self.min = values[0]
                self.max = values[0]
            
        if is_arg(f'{self.name}_dist', args):
            dist = get_arg(f'{self.name}_dist', self.dist, args)
            if not isinstance(self.dist, list):
                self.dist = dist
            else:
                self.dist = dist[0]
                if len(dist) > 1:
                    self.dist_args = [ float(v) for v in dist[1:] ]
                else:
                    self.dist_args = None
import numpy as np
from scipy.interpolate import interp1d

from ..util.args import *
from ..util.dist import RANDOM_DISTS

class GridAxis():
    def __init__(self, name, values=None, order=None, orig=None):

        if not isinstance(orig, GridAxis):
            self.name = name
            self.values = values
            self.order = order
            self.index = None
            self.min = None
            self.max = None
            self.dist = None
            self.dist_args = None
            self.ip_to_index = None
            self.ip_to_value = None
        else:
            self.name = name or orig.name
            self.values = values if values is not None else orig.values
            self.order = order if order is not None else orig.order
            self.index = orig.index
            self.min = orig.min
            self.max = orig.max
            self.dist = orig.dist
            self.dist_args = orig.dist_args
            self.ip_to_index = orig.ip_to_index
            self.ip_to_value = orig.ip_to_value

    def add_args(self, parser):
        parser.add_argument(f'--{self.name}', type=float, nargs='*', default=None, help=f'Limit on {self.name}.\n')
        parser.add_argument(f'--{self.name}-dist', type=str, nargs='*', help=f'Distribution type and params for {self.name}.\n')

    def init_from_args(self, args, mode='minmax'):
        if is_arg(self.name, args):
            values = args[self.name]

            if mode == 'minmax':
                if len(values) >= 2:
                    self.min = values[0]
                    self.max = values[1]
                else:
                    self.min = values[0]
                    self.max = values[0]
            elif mode == 'aux':
                # TODO: add log sampling?
                self.values = np.linspace(*values)
                self.build_index()
            else:
                raise NotImplementedError()

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
              
    def build_index(self):
        # NOTE: assume one dimension here
        self.index = {v: i[0] for i, v in np.ndenumerate(self.values)}
        self.min = np.min(self.values)
        self.max = np.max(self.values)
        if self.values.shape[0] > 1:
            self.ip_to_index = interp1d(self.values, np.arange(self.values.shape[0]), fill_value='extrapolate')
            self.ip_to_value = interp1d(np.arange(self.values.shape[0]), self.values, fill_value='extrapolate')
        else:
            self.ip_to_index = None
            self.ip_to_value = None

    def get_index(self, value):
        return self.index[value]

    def get_nearest_index(self, value):
        return np.abs(self.values - value).argmin()

    def get_nearby_indexes(self, value):
        i1 = self.get_nearest_index(value)
        if value < self.values[i1]:
            i1, i2 = i1 - 1, i1
        else:
            i1, i2 = i1, i1 + 1

        # Verify if indexes inside bounds
        if i1 < 0 or i2 >= self.values.shape[0]:
            return None
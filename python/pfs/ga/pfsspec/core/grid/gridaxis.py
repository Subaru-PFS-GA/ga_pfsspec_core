import numpy as np
from scipy.interpolate import interp1d

from ..util.args import *
from ..sampling.parameter import Parameter

class GridAxis(Parameter):
    def __init__(self, name, values=None, order=None, orig=None):
        super().__init__(name, orig=orig)

        if not isinstance(orig, GridAxis):
            self.values = values
            self.order = order
            self.index = None
            self.ip_to_index = None
            self.ip_to_value = None
        else:
            self.values = values if values is not None else orig.values
            self.order = order if order is not None else orig.order
            self.index = orig.index
            self.ip_to_index = orig.ip_to_index
            self.ip_to_value = orig.ip_to_value

    def init_from_args(self, args):
        super().init_from_args(args)
        self.build_index()

        # TODO: if this is an auxiliary axis in a grid sampler
        #       we need to generate a value grid
        # if is_arg(self.name, args):
        #     values = args[self.name]
        #     if len(values) <= 2:
        #         self.values = np.array(values)
        #     elif len(values) > 2:
        #         self.values = np.linspace(*values)

    def build_index(self):
        # NOTE: assume one dimension here
        self.index = {v: i[0] for i, v in np.ndenumerate(self.values)}

        # If limits are not otherwise set, use maximum extents
        if self.min is None:
            self.min = np.min(self.values)

        if self.max is None:
            self.max = np.max(self.values)

        # Interpolators to go from index to value and vice versa
        if self.values.shape[0] > 1:
            self.ip_to_index = interp1d(self.values, np.arange(self.values.shape[0]), fill_value='extrapolate', assume_sorted=True)
            self.ip_to_value = interp1d(np.arange(self.values.shape[0]), self.values, fill_value='extrapolate', assume_sorted=True)
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
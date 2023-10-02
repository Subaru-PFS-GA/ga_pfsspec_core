import numpy as np

from pfs.ga.pfsspec.core.util.args import *
from pfs.ga.pfsspec.core import PfsObject

class ParameterSampler(PfsObject):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, ParameterSampler):
            self.parameters = {}
            self.match_params = None            # Match parameters from this dataframe
            self.sample_count = 0               # Number of samples to generate
                                                # TODO: move this elsewhere?
        else:
            self.parameters = orig.parameters
            self.match_params = orig.match_params
            self.sample_count = orig.sample_count

    def enumerate_parameters(self):
        for i, (k, par) in enumerate(self.parameters.items()):
            yield i, k, par

    def add_parameter(self, parameter):
        self.parameters[parameter.name] = parameter

    def add_args(self, parser):
        parser.add_argument('--sample-count', type=int, default=None, help='Number of samples to be generated.\n')

        # Additional axes
        for i, k, par in self.enumerate_parameters():
            par.add_args(parser)

    def init_from_args(self, config, args):
        self.sample_count = get_arg('sample_count', self.sample_count, args)

        # Register command-line arguments for auxiliary axes
        for i, k, par in self.enumerate_parameters():
            # Using aux mode because we need to generate the axis values from the
            # parameters passed to via args.
            par.init_from_args(args)

    def draw_random_param(self, par):
        if par.dist == 'const':
            r = par.value
        elif par.dist == 'int':
            r = np.random.randint(par.min, par.max)
        elif par.dist == 'choice':
            r = np.random.choice(par.dist_args).item()
        else:
            dist = par.get_dist(random_state=self.random_state)

            if dist is not None:
                r = dist.sample()
            else:
                r = None
            
        return r

    def draw_random_params(self, params=None):
        """
        Draw random values for each of the axes, those of the grid as well as
        auxiliary. If the distribution is not specified (and min and max are None),
        do not sample the value.
        """
        
        params = params or {}

        for i, k, par in self.enumerate_parameters():
            v = self.draw_random_param(par)
            if v is not None:
                params[k] = self.draw_random_param(par)

        return params
    
    def get_matched_params(self, i):
        # When ˙match_params` is set to a DataFrame, we need to look up the ith
        # record and return the row as a dictionary

        # TODO: this is now using pandas, consider making it more generic

        params = self.match_params[self.match_params['id'] == i].to_dict('records')[0]
        return params

    def sample_params(self, i):
        if self.match_params is not None:
            params = self.get_matched_params(i)
        else:
            params = self.draw_random_params()

        return params
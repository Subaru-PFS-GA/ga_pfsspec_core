from .parameter import Parameter
from .parametersampler import ParameterSampler

from .distribution import Distribution
from .compositedistribution import CompositeDistribution
from .uniformdistribution import UniformDistribution
from .normaldistribution import NormalDistribution
from .lognormaldistribution import LogNormalDistribution
from .betadistribution import BetaDistribution

DISTRIBUTIONS = {
    'composite': CompositeDistribution,
    'uniform': UniformDistribution,
    'beta': BetaDistribution,
    'normal': NormalDistribution,
    'lognormal': LogNormalDistribution
}
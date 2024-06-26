from .binning import Binning
from .fluxconservingresampler import FluxConservingResampler
from .interp1dresampler import Interp1dResampler
from .pysynphotresampler import PysynphotResampler
from .specutilsresampler import SpecutilsResampler

RESAMPLERS = {
    'interp': Interp1dResampler,
    'fluxcons': FluxConservingResampler,
}
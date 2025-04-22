from .meansnr import MeanSnr
from .mediansnr import MedianSnr
from .quantilesnr import QuantileSnr
from .stoehrsnr import StoehrSnr

SNR_TYPES = {
    'mean': MeanSnr,
    'median': MedianSnr,
    'quantile': QuantileSnr,
    'stoehr': StoehrSnr
}
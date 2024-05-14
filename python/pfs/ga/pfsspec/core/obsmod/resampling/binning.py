import numpy as np

class Binning():
    
    @staticmethod
    def find_wave_edges(wave, binning='lin'):
        if binning == 'lin':
            wave_edges = np.empty((wave.shape[0] + 1,), dtype=wave.dtype)
            wave_edges[1:-1] = 0.5 * (wave[1:] + wave[:-1])
            wave_edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
            wave_edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
        elif binning == 'log':
            raise NotImplementedError()

        return wave_edges
    
    @staticmethod
    def round_wave_min_max(wmin, wmax, binsize, binning='lin'):
        if binning == 'lin':
            wmin = binsize * np.floor(wmin / binsize)
            wmax = binsize * np.ceil(wmax / binsize)
        elif binning == 'log':
            wmin = np.exp(binsize * np.floor(np.log(wmin) / binsize))
            wmax = np.exp(binsize * np.ceil(np.log(wmax) / binsize))
        else:
            raise NotImplementedError()

        return wmin, wmax
    
    @staticmethod
    def generate_wave_bins(wmin, wmax, nbins=None, binsize=None, binning='lin'):
        if nbins is not None:
            if binning == 'lin':
                wave_edges = np.linspace(wmin, wmax, nbins + 1)
            elif binning == 'log':
                wave_edges = np.exp(np.linspace(np.log(wmin), np.log(wmax), nbins + 1))
            else:
                raise NotImplementedError()
        elif binsize is not None:
            if binning == 'lin':
                wave_edges = np.arange(wmin, wmax + binsize, binsize)
            elif binning == 'log':
                wave_edges = np.exp(np.arange(np.log(wmin), np.log(wmax) + binsize, binsize))
        else:
            raise ValueError('Either `nbins` or `binsize` must be specified.')
        
        if binning == 'lin':
            wave = 0.5 * (wave_edges[1:] + wave_edges[:-1])
        elif binning == 'log':
            wave = np.exp(0.5 * (np.log(wave_edges[1:]) + np.log(wave_edges[:-1])))
        else:
            raise NotImplementedError()
        
        return wave, wave_edges
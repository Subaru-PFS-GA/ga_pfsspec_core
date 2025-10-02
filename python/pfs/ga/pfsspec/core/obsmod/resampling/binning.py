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
    def generate_wave_bins(wmin, wmax, nbins=None, binsize=None, resolution=None, binning='lin', align='edges'):
        if nbins is not None:
            if align == 'edges':
                if binning == 'lin':
                    wave_edges = np.linspace(wmin, wmax, nbins + 1)
                    wave = 0.5 * (wave_edges[1:] + wave_edges[:-1])
                elif binning == 'log':
                    log_wave_edges = np.linspace(np.log(wmin), np.log(wmax), nbins + 1)
                    log_wave = 0.5 * (log_wave_edges[1:] + log_wave_edges[:-1])
                else:
                    raise NotImplementedError()
            elif align == 'center':
                if binning == 'lin':
                    wave = np.linspace(wmin, wmax, nbins)
                    binsize = (wave[-1] - wave[0]) / (nbins - 1)
                    wave_edges = np.linspace(wave[0] - 0.5 * binsize, wave[-1] + 0.5 * binsize, nbins + 1)
                elif binning == 'log':
                    log_wave = np.linspace(np.log(wmin), np.log(wmax), nbins)
                    binsize = (log_wave[-1] - log_wave[0]) / (nbins - 1)
                    log_wave_edges = np.linspace(log_wave[0] - 0.5 * binsize, log_wave[-1] + 0.5 * binsize, nbins + 1)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif binsize is not None:
            if align == 'edges':
                if binning == 'lin':
                    wave_edges = np.arange(wmin, wmax + binsize, binsize)
                    wave = 0.5 * (wave_edges[1:] + wave_edges[:-1])
                elif binning == 'log':
                    log_wave_edges = np.arange(np.log(wmin), np.log(wmax) + binsize, binsize)
                    log_wave = 0.5 * (log_wave_edges[1:] + log_wave_edges[:-1])
                else:
                    raise NotImplementedError()
            elif align == 'center':
                if binning == 'lin':
                    wave = np.arange(wmin, wmax + binsize, binsize)
                    wave_edges = np.arange(wave[0] - 0.5 * binsize, wave[-1] + 1.5 * binsize, binsize)
                elif binning == 'log':
                    log_wave = np.arange(np.log(wmin), np.log(wmax) + binsize, binsize)
                    log_wave_edges = np.arange(log_wave[0] - 0.5 * binsize, log_wave[-1] + 0.5 * binsize, binsize)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif resolution is not None and binning == 'log':
            if align == 'edges':
                nbins = int(np.log(wmax / wmin) / np.log(1 + 1 / resolution))
                log_wave_edges = np.linspace(np.log(wmin), np.log(wmax), nbins + 1)
                log_wave = 0.5 * (log_wave_edges[1:] + log_wave_edges[:-1])
            elif align == 'center':
                binsize = np.log(1 + 1 / resolution)
                log_wave = np.arange(np.log(wmin), np.log(wmax) + binsize, binsize)
                log_wave_edges = np.arange(log_wave[0] - 0.5 * binsize, log_wave[-1] + 1.5 * binsize, binsize)
            else:
                raise NotImplementedError()
        else:
            raise ValueError('Either `nbins` or `binsize` must be specified.')
        
        if binning == 'lin':
            pass
        elif binning == 'log':
            wave = np.exp(log_wave)
            wave_edges = np.exp(log_wave_edges)
        else:
            raise NotImplementedError()
        
        return wave, wave_edges
    
    @staticmethod
    def get_wave_edges_1d(wave_edges):
        if wave_edges is None:
            return None
        elif wave_edges.ndim == 1:
            return wave_edges
        elif wave_edges.ndim == 2:
            # TODO: this assumes that the bind are adjecent!
            return np.concatenate([wave_edges[0], wave_edges[1][-1:]])
        else:
            raise NotImplementedError()

    @staticmethod
    def get_wave_edges_2d(wave_edges):
        if wave_edges is None:
            return None
        elif wave_edges.ndim == 1:
            return np.stack([wave_edges[:-1], [wave_edges[1:]]], axis=0)
        elif wave_edges.ndim == 2:
            return wave_edges
        else:
            raise NotImplementedError()
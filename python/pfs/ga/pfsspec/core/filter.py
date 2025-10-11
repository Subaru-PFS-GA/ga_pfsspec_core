import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from .constants import Constants
from .physics import Physics
from .pfsobject import PfsObject
from .util.copy import safe_deep_copy

class Filter(PfsObject):
    def __init__(self, name=None, orig=None):
        super(Filter, self).__init__()
        
        if not isinstance(orig, Filter):
            self.name = name if name is not None else '(unnamed)'
            self.wave = None
            self.thru = None

            self.ip = None
            self.wave_eff = None
        else:
            self.name = name if name is not None else orig.name
            self.wave = safe_deep_copy(orig.wave)
            self.thru = safe_deep_copy(orig.thru)
        
            self.wave_eff = safe_deep_copy(orig.wave_eff)

    def read(self, filename):
        self.filename = filename
        self.name = os.path.splitext(os.path.basename(filename))[0]
        
        [wave, thru] = np.loadtxt(filename).transpose()
        ix = np.argsort(wave)
        self.wave = wave[ix]
        self.thru = thru[ix]

    def calculate_wave_eff(self):
        """
        Calculate the effective wavelength of the filter.
        """

        if self.wave is None or self.thru is None:
            raise ValueError('Filter data not loaded.')

        num = np.trapezoid(self.wave * self.thru, self.wave)
        den = np.trapezoid(self.thru, self.wave)

        self.wave_eff = num / den

        return self.wave_eff

    def trim(self, tol=1e-5):
        """
        Trim the filter to the wavelength range where the throughput is above `tol`.
        """

        mask = self.thru > tol
        fr = np.min(np.where(mask)[0])
        to = np.max(np.where(mask)[0]) + 1
        self.wave = self.wave[fr:to]
        self.thru = self.thru[fr:to]

    def extend(self, min, max, res):
        wavemin = self.wave.min()
        wavemax = self.wave.max()
        if (min < wavemin):
            w = np.arange(min, wavemin, res)
            self.wave = np.hstack((w, self.wave))
            self.thru = np.hstack((np.zeros(len(w)), self.thru))

        if (max > wavemax):
            w = np.arange(wavemax, max, res) + res
            self.wave = np.hstack((self.wave, w))
            self.thru = np.hstack((self.thru, np.zeros(len(w))))

    def plot(self, ax=None, xlim=Constants.DEFAULT_PLOT_WAVE_RANGE, ylim=None, labels=True):
        ax = self.plot_getax(ax, xlim, ylim)
        ax.plot(self.wave, self.thru, label=self.name)
        if labels:
            ax.set_xlabel(r'$\lambda$ [A]')
            ax.set_ylabel(r'$R_\lambda$')

        return ax

    def interpolate(self):
        if self.ip is None:
            self.ip = interp1d(self.wave, self.thru, kind='linear', bounds_error=False, fill_value=0.0)
    
    def synth_flux(self, wave, flux):
        """
        Integrate the flux through the filter. Assume that we have flux density
        per unit wavelength and we are calculating AB magnitudes.

        Parameters
        ----------
        wave : array-like
            Wavelength array of the flux.
        flux : array-like
            Flux array to integrate.

        Returns
        -------
        float
            Integrated flux through the filter.
        """

        self.interpolate()
        flux = Physics.synth_flux_lam(wave, flux, self.ip(wave))
        return flux
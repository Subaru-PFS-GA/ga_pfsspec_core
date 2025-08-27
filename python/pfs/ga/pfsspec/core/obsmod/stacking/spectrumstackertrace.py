import os
import numpy as np

from pfs.ga.pfsspec.core.plotting import SpectrumPlot, DistributionPlot, styles
from pfs.ga.pfsspec.core import Trace, SpectrumTrace
from pfs.ga.pfsspec.core.util.args import *

class SpectrumStackerTrace(Trace, SpectrumTrace):

    def __init__(self,
                 id=None,
                 figdir='.', logdir='.',
                 plot_inline=False, 
                 plot_level=Trace.PLOT_LEVEL_NONE, 
                 log_level=Trace.LOG_LEVEL_NONE):
        
        super().__init__(id=id,
                         figdir=figdir, logdir=logdir,
                         plot_inline=plot_inline, 
                         plot_level=plot_level,
                         log_level=log_level)

        self.__id = id                          # Identity represented as string

        self.plot_input = None
        self.plot_stack = None
        self.plot_merged = None
        self.plot_fit_spec = {}

        self.reset()

    #region Properties

    def __get_id(self):
        return self.__id
    
    def __set_id(self, value):
        self.__id = value

    id = property(__get_id, __set_id)

    #endregion

    def reset(self):
        pass

    def update(self, figdir=None, logdir=None, id=None):
        super().update(figdir=figdir, logdir=logdir)
        
        self.__id = id if id is not None else self.__id

    def add_args(self, config, parser):
        super().add_args(config, parser)
    
    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

        self.plot_input = get_arg('plot_input', self.plot_input, args)
        self.plot_stack = get_arg('plot_stack', self.plot_stack, args)
        self.plot_merged = get_arg('plot_merged', self.plot_merged, args)
        self.plot_fit_spec = get_arg('plot_fit_spec', self.plot_fit_spec, args)

    def on_coadd_start_stack(self, spectra, no_continuum_bit, exclude_bits):
        """
        Fired when the coaddition process starts.
        
        Parameters
        ----------
        spectra : dict
            The dictionary of spectra, the continuum model or the flux correction attached
            but not applied yet.
        no_continuum_bit : int
            The bit mask for the no continuum flag.
        exclude_bits : list
            The list of bit masks to exclude from coadding.
        """

        if self.plot_input is None and self.plot_level >= Trace.PLOT_LEVEL_DEBUG \
            or self.plot_input:

            self._plot_spectra('pfsGA-coadd-input-{id}',
                               spectra,
                               plot_flux=True, plot_flux_err=True, plot_mask=True,
                               mask_bits=no_continuum_bit | exclude_bits,
                               normalize_cont=True, apply_flux_corr=True,
                               title='Coadd input spectra - {id}',
                               nrows=2, ncols=1, diagram_size=(6.5, 3.5))

            self.flush_figures()

    def on_coadd_finish_stack(self, coadd_spectra, no_continuum_bit, exclude_bits):
        """
        Fired when the coaddition process finishes.
        
        Parameters
        ----------
        spectra : dict
            The dictionary of spectra, the continuum model or the flux correction attached
            but not applied yet.
        no_continuum_bit : int
            The bit mask for the no continuum flag.
        exclude_bits : list
            The list of bit masks to exclude from coadding.
        """

        if self.plot_stack is None and self.plot_level >= Trace.PLOT_LEVEL_INFO \
            or self.plot_stack:

            self._plot_spectra('pfsGA-coadd-stack-{id}',
                                coadd_spectra,
                                plot_flux=True, plot_flux_err=True, plot_mask=True,
                                mask_bits=no_continuum_bit | exclude_bits,
                                normalize_cont=False, apply_flux_corr=False,
                                title='Coadd stacked spectrum - {id}',
                                diagram_size=(6.5, 3.5))

            self.flush_figures()
    
    def on_coadd_finish_merged(self, merged_spectrum):
        """
        Fired when the coaddition process finishes and the merged spectrum is created.
        
        Parameters
        ----------
        merged_spectrum : PfsSpectrum
            The merged spectrum.
        """

        if self.plot_merged is None and self.plot_level >= Trace.PLOT_LEVEL_INFO \
            or self.plot_merged:

            self._plot_spectrum('pfsGA-coadd-merged-{id}',
                                arm='all',
                                spectrum=merged_spectrum,
                                plot_flux=True, plot_flux_err=True, plot_mask=True,
                                normalize_cont=False, apply_flux_corr=False,
                                title='Coadd merged spectrum - {id}',
                                diagram_size=(6.5, 3.5))

            self.flush_figures()
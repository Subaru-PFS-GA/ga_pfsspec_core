import os
import numpy as np

from pfs.ga.pfsspec.core.plotting import SpectrumPlot, DistributionPlot, styles

class SpectrumTrace():
    """
    Mixin to provide trace functionality to Spectrum processing trace objects.
    """

    def __init__(self):
        pass

    def add_args(self, config, parser):
        pass

    def init_from_args(self, script, config, args):
        pass

    def _plot_spectrum(self,
                       key,
                       arm=None,
                       spectrum=None,
                       template=None,
                       apply_flux_corr=None,
                       normalize_cont=False,
                       plot_flux=None, plot_flux_err=None,
                       plot_continuum=None,
                       plot_template=True,
                       plot_residual=False,
                       plot_mask=False, mask_bits=None,
                       wave_include=None, wave_exclude=None,
                       wlim=None, auto_limits=True,
                       title=None,
                       diagram_size=(6.5, 2.0)):
        
        """
        Plot a single spectrum, optinally along with a template.
        """

        f = self.get_diagram_page(key, 1, 1, 1,
                                  title=title,
                                  diagram_size=diagram_size)
        
        flux_calibrated = None
        
        if spectrum is not None and spectrum.is_flux_calibrated is not None and not spectrum.is_flux_calibrated:
            flux_calibrated = False
        else:
            flux_calibrated = True

        if template is not None and template.is_flux_calibrated is not None and not template.is_flux_calibrated:
            flux_calibrated = False
        else:
            flux_calibrated = True
        
        p, ax = self.__create_spectrum_plot(f, 0, 0, 0,
                                            plot_flux=plot_flux,
                                            plot_mask=plot_mask,
                                            plot_flux_err=plot_flux_err,
                                            plot_continuum=plot_continuum,
                                            flux_calibrated=flux_calibrated)

        if spectrum is not None:
            self.__plot_spectrum_impl(p, arm, 
                                      spectrum,
                                      apply_flux_corr,
                                      normalize_cont,
                                      plot_flux, plot_flux_err,
                                      plot_continuum,
                                      plot_mask, mask_bits,
                                      wlim, auto_limits)
            
        if template is not None:
            self.__plot_template_impl(p, arm,
                                      template,
                                      apply_flux_corr,
                                      normalize_cont,
                                      plot_mask, mask_bits,
                                      wlim, auto_limits)
            
        if plot_residual and spectrum is not None and template is not None:
            self.__plot_residual_impl(p, arm,
                                      spectrum, template,
                                      apply_flux_corr,
                                      normalize_cont,
                                      plot_mask, mask_bits,
                                      wlim, auto_limits)

        # TODO: Add SNR, exp time, obs date
        if spectrum is not None:
            p.title = spectrum.get_name()
        elif template is not None:
            p.title = template.get_name()

        if wave_include is not None:
            p.shade_wave_ranges(wave_include, facecolor='green')

        if wave_exclude is not None:
            p.shade_wave_ranges(wave_exclude, facecolor='red')

        p.apply()

        f.match_limits()
        
    def _plot_spectra(self,
                      key,
                      spectra=None,
                      templates=None,
                      apply_flux_corr=None,
                      normalize_cont=False,
                      plot_flux=None, plot_flux_err=None,
                      plot_continuum=None,
                      plot_template=None,
                      plot_residual=None,
                      plot_mask=None, mask_bits=None,
                      wave_include=None, wave_exclude=None,
                      wlim=None, auto_limits=True,
                      title=None,
                      nrows=4, ncols=1, diagram_size=(6.5, 2.0)):
        
        """
        Plot a set of spectra, each in its own plot arranged in a grid format.

        Parameters
        ----------
        key : str
            Key to identify the diagram page.
        spectra : dict of Spectrum or dict of list of Spectrum
            Dictionary of spectra, where the keys are the arms and the values are the spectra.
        apply_flux_corr : dict of list of bool
            Whether to apply the flux correction model
        normalize_cont : bool
            If available, normalize the flux with the continuum.
        plot_flux : bool
            Plot the spectrum.
        plot_flux_err : bool
            Plot the flux errors.
        plot_continuum : bool
            Plot the continuum.
        plot_mask : bool
            Plot the mask.
        mask_bits : list of str
            List of mask bits to plot.
        wave_include : list of list of float
            List of wavelength ranges to include in the mask. Will be plotted as shaded regions.
        wave_exclude : list of list of float
            List of wavelength ranges to exclude from the mask. Will be plotted as shaded regions.
        wlim : list of float
            Wavelength limits to plot.
        auto_limits : bool
            Automatically set the limits.
        title : str
            Title of the diagram.
        nrows : int
            Number of rows in the grid.
        ncols : int
            Number of columns in the grid.
        diagram_size : tuple of float
            Size of the diagram.
        """

        def get_plot(p):
            if p is None:
                if spectrum.is_flux_calibrated is not None and not spectrum.is_flux_calibrated:
                    flux_calibrated = False
                else:
                    flux_calibrated = True

                p, ax = self.__create_spectrum_plot(f, j, k, l,
                                                    plot_flux=plot_flux,
                                                    plot_mask=plot_mask,
                                                    plot_flux_err=plot_flux_err,
                                                    plot_continuum=plot_continuum,
                                                    flux_calibrated=flux_calibrated)
                
            return p
        
        # Number of exposures
        nexp = np.max([ len(spectra[arm]) for arm in spectra.keys() ])

        npages = int(np.ceil(nexp / (ncols * nrows)))
        f = self.get_diagram_page(key, npages, nrows, ncols,
                                  title=title,
                                  diagram_size=diagram_size)
       
        for i, (j, k, l) in enumerate(np.ndindex((npages, nrows, ncols))):
            if i == nexp:
                break

            p, ax = None, None

            if spectra is not None:
                arms = spectra.keys()
            elif templates is not None:
                arms = templates.keys()
            else:
                raise ValueError("No spectra or templates provided")
            
            arms = list(arms)

            for arm in arms:
                # Spectra are either dict of lists or dict of dicts
                if spectra is None:
                    spectrum = None
                elif isinstance(spectra[arm], dict):
                    keys = list(sorted(spectra[arm].keys()))
                    if i < len(keys):
                        spectrum = spectra[arm][keys[i]]
                    else:
                        spectrum = None
                elif isinstance(spectra[arm], list):
                    if i < len(spectra[arm]):
                        spectrum = spectra[arm][i]
                    else:
                        spectrum = None
                else:
                    raise NotImplementedError()
                
                if templates is None:
                    template = None
                elif isinstance(templates[arm], dict):
                    keys = list(sorted(templates[arm].keys()))
                    if i < len(keys):
                        template = templates[arm][keys[i]]
                    else:
                        template = None
                elif isinstance(templates[arm], list):
                    if i < len(templates[arm]):
                        template = templates[arm][i]
                    else:
                        template = None
                else:
                    raise NotImplementedError()
                
                # Initialize the plot when plotting the first spectrum
                if spectrum is not None:
                    p = get_plot(p)
                    self.__plot_spectrum_impl(p, arm, 
                                              spectrum,
                                              apply_flux_corr,
                                              normalize_cont,
                                              plot_flux, plot_flux_err, plot_continuum,
                                              plot_mask, mask_bits,
                                              wlim, auto_limits)
                    
                    p.title = spectrum.get_name()
                        
                if plot_template and template is not None:
                    p = get_plot(p)
                    self.__plot_template_impl(p, arm,
                                              template,
                                              apply_flux_corr,
                                              normalize_cont,
                                              plot_mask, mask_bits,
                                              wlim, auto_limits)
                    
                if plot_residual and spectrum is not None and template is not None:
                    p = get_plot(p)
                    self.__plot_residual_impl(p, arm,
                                              spectrum, template,
                                              apply_flux_corr,
                                              normalize_cont,
                                              plot_mask, mask_bits,
                                              wlim, auto_limits)

            if p is not None:
                if wave_include is not None:
                    p.shade_wave_ranges(wave_include, facecolor='green')

                if wave_exclude is not None:
                    p.shade_wave_ranges(wave_exclude, facecolor='red')

                p.apply()

        f.match_limits()

    def __create_spectrum_plot(self, f, j, k, l,
                               plot_mask, plot_flux, plot_flux_err, plot_continuum,
                               flux_calibrated=True,
                               title=None):
        
        p = SpectrumPlot(flux_calibrated=flux_calibrated)
        ax = f.add_diagram((j, k, l), p)

        p.plot_mask = plot_mask
        p.plot_flux = plot_flux
        p.plot_flux_err = plot_flux_err
        p.plot_cont = plot_continuum
        p.title = title

        return p, ax

    def __plot_spectrum_impl(self, p, arm,
                             spectrum,
                             apply_flux_corr,
                             normalize_cont,
                             plot_flux, plot_flux_err,
                             plot_continuum,
                             plot_mask, mask_bits,
                             wlim, auto_limits):
        
        # TODO: define arm color in styles
        if spectrum is not None and (plot_flux or plot_flux_err or plot_continuum or plot_mask):
            p.plot_spectrum(spectrum,
                            apply_flux_corr=apply_flux_corr,
                            normalize_cont=normalize_cont,
                            plot_flux=plot_flux, plot_flux_err=plot_flux_err,
                            plot_mask=plot_mask, mask_bits=mask_bits,
                            wlim=wlim, auto_limits=auto_limits)

    def __plot_template_impl(self, p, arm,
                             template,
                             apply_flux_corr,
                             normalize_cont,
                             plot_mask, mask_bits,
                             wlim, auto_limits):
        
        if template is not None:
            p.plot_template(template,
                            apply_flux_corr=apply_flux_corr,
                            normalize_cont=normalize_cont,
                            plot_mask=plot_mask, mask_bits=mask_bits,
                            wlim=wlim)

    def __plot_residual_impl(self, p, arm,
                              spectrum, template,
                              apply_flux_correction,
                              normalize_cont,
                              plot_mask, mask_bits,
                              wlim, auto_limits):
        
        if spectrum is not None and template is not None:
            p.plot_residual(spectrum, template,
                            apply_flux_correction=apply_flux_correction,
                            plot_mask=plot_mask, mask_bits=mask_bits,
                            wlim=wlim, auto_limits=auto_limits)

    def _save_spectrum_history(self, filename, spectrum):
        """
        Save the processing history of a spectrum
        """
        if spectrum.history is not None:

            filename = filename.format(id=self.id)

            fn = os.path.join(self.logdir, filename)
            self.make_outdir(fn)
            spectrum.history.save(fn)

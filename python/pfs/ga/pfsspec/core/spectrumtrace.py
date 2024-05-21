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

    def _plot_spectrum(self, key, arm=None,
                       spectrum=None, processed_spectrum=None,
                       template=None, processed_template=None,
                       plot_spectrum=True, plot_flux_err=True, plot_processed_spectrum=True,
                       plot_template=True, plot_processed_template=True,
                       plot_residuals=False, plot_continuum=False,
                       plot_mask=False, mask_bits=None,
                       wlim=None, auto_limits=True,
                       title=None):
        
        """
        Plot a single spectrum, optinally along with a template.
        """

        f = self.get_diagram_page(key, 1, 1, 1,
                                  title=title,
                                  diagram_size=(6.5, 2.0))
        
        p, ax = self.__create_spectrum_plot(f, 0, 0, 0,
                                            plot_mask=plot_mask,
                                            plot_flux_err=plot_flux_err,
                                            plot_continuum=plot_continuum)

        self.__plot_spectrum_impl(p, arm, 
                                  spectrum, processed_spectrum,
                                  template, processed_template,
                                  plot_spectrum, plot_flux_err, plot_processed_spectrum,
                                  plot_template, plot_processed_template,
                                  plot_residuals, plot_continuum,
                                  plot_mask=plot_mask, mask_bits=mask_bits,
                                  wlim=wlim, auto_limits=auto_limits)

        # TODO: Add SNR, exp time, obs date
        if spectrum is not None:
            p.title = spectrum.get_name()
        elif processed_spectrum is not None:
            p.title = processed_spectrum.get_name()
        elif template is not None:
            p.title = template.get_name()
        elif processed_template is not None:
            p.title = processed_template.get_name()

        p.apply()

        f.match_limits()
        
    def _plot_spectra(self, key,
                      spectra=None, processed_spectra=None,
                      templates=None, processed_templates=None,
                      plot_spectrum=True, plot_flux_err=True, plot_processed_spectrum=True,
                      plot_template=True, plot_processed_template=True,
                      plot_residuals=False, plot_continuum=False,
                      plot_mask=False, mask_bits=None,
                      wlim=None, auto_limits=True,
                      title=None,
                      nrows=4, ncols=1, diagram_size=(6.5, 2.0)):
        
        """
        Plot a set of spectra in a grid format.
        """
        
        # Number of exposures
        nexp = np.max([ len(spectra[arm]) for arm in spectra.keys() ])

        npages = int(np.ceil(nexp / (ncols * nrows)))
        f = self.get_diagram_page(key, npages, nrows, ncols,
                                  title=title,
                                  diagram_size=diagram_size)
       
        for i, (j, k, l) in enumerate(np.ndindex((npages, nrows, ncols))):
            if i == nexp:
                break

            p, ax = self.__create_spectrum_plot(f, j, k, l,
                                                plot_mask=plot_mask,
                                                plot_flux_err=plot_flux_err,
                                                plot_continuum=plot_continuum)

            if spectra is not None:
                arms = spectra.keys()
            elif processed_spectra is not None:
                arms = processed_spectra.keys()
            elif templates is not None:
                arms = templates.keys()
            elif processed_templates is not None:
                arms = processed_templates.keys()
            else:
                raise ValueError("No spectra or templates provided")
            
            arms = list(arms)

            for arm in arms:
                spectrum = spectra[arm][i] if spectra is not None else None

                self.__plot_spectrum_impl(p, arm, 
                                          spectrum,
                                          processed_spectra[arm][i] if processed_spectra is not None else None,
                                          templates[arm] if templates is not None else None,
                                          processed_templates[arm][i] if processed_templates is not None else None,
                                          plot_spectrum, plot_flux_err, plot_processed_spectrum,
                                          plot_template, plot_processed_template,
                                          plot_residuals, plot_continuum,
                                          plot_mask=plot_mask, mask_bits=mask_bits,
                                          wlim=wlim, auto_limits=auto_limits)

                if spectrum is not None:
                    p.title = spectrum.get_name()

                p.apply()

        f.match_limits()

    def __create_spectrum_plot(self, f, j, k, l,
                               plot_mask, plot_flux_err, plot_continuum,
                               title=None):
        p = SpectrumPlot()
        ax = f.add_diagram((j, k, l), p)

        p.plot_mask = plot_mask
        p.plot_flux_err = plot_flux_err
        p.plot_cont = plot_continuum
        p.title = title

        return p, ax

    def __plot_spectrum_impl(self, p, arm,
                             spectrum, processed_spectrum,
                             template, processed_template,
                             plot_spectrum, plot_flux_err, plot_processed_spectrum,
                             plot_template, plot_processed_template,
                             plot_residuals, plot_continuum,
                             plot_mask, mask_bits,
                             wlim, auto_limits):
        
        # TODO: define arm color in styles
        if plot_spectrum and spectrum is not None:
            p.plot_spectrum(spectrum, plot_flux_err=plot_flux_err,
                            plot_mask=plot_mask, mask_bits=mask_bits,
                            wlim=wlim, auto_limits=auto_limits)

        if plot_processed_spectrum and processed_spectrum is not None:
            p.plot_spectrum(processed_spectrum, plot_flux_err=plot_flux_err,
                            plot_mask=plot_mask, mask_bits=mask_bits,
                            wlim=wlim, auto_limits=auto_limits)

        if plot_template and template is not None:
            p.plot_spectrum(template, wlim=wlim, auto_limits=auto_limits)

        if plot_processed_template and processed_template is not None:
            p.plot_template(processed_template, wlim=wlim)

        if plot_residuals and processed_template is not None:
            temp = processed_template
            p.plot_residual(spectrum, temp, wlim=wlim)

    def _save_spectrum_history(self, filename, spectrum):
        """
        Save the processing history of a spectrum
        """
        if spectrum.history is not None:

            filename = filename.format(id=self.id)

            fn = os.path.join(self.logdir, filename)
            self.make_outdir(fn)
            with open(fn, "w") as f:
                f.writelines([ s + '\n' for s in spectrum.history ])
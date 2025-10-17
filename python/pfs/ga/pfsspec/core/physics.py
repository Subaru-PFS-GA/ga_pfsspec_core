import numpy as np
from scipy.integrate import trapezoid

class Physics():
    """
    Class containing physical constants and utility functions.
    
    Some default units:
    - wavelength: angstrom
    - flam: flux density per unit wavelength: erg/s/cm^2/Angstrom
    - fnu: flux density per unit frequency: erg/s/cm^2/Hz
    """


    h = 6.62607015e-34  # J s
    c = 299792458       # m/s
    k_B = 1.380649e-23  # J/K

    HYDROGEN_LIMITS = [3646.0, 8203.6, 14584]                   # all numbers in air

    @staticmethod
    def planck(wave, T):
        a = 2.0 * Physics.h * Physics.c**2                      # J m+2 s-1
        b = Physics.h * Physics.c / (wave * Physics.k_B * T)    # 1
        intensity = a / (wave**5 * (np.exp(b) - 1.0))           # J m-2 m-1 s-1
        return intensity

    @staticmethod
    def angstrom_to_nm(wave):
        if wave is not None:
            return 1e-1 * wave
        else:
            return None
        
    @staticmethod
    def nm_to_angstrom(wave):
        if wave is not None:
            return 1e1 * wave
        else:
            return None
    
    @staticmethod
    def jy_to_abmag(flux):
        if flux is not None:
            return -2.5 * np.log10(flux) + 8.90
        else:
            return None
    
    @staticmethod
    def abmag_to_jy(mag):
        if mag is not None:
            return 1e23 * 10 ** (-0.4 * (mag + 48.6))
        else:
            return None

    @staticmethod
    def fnu_to_flam(wave, fnu):
        # ergs/cm**2/s/Hz to erg/s/cm^2/A
        if wave is not None and fnu is not None:
            flam = fnu / (3.336e-19 * wave**2)          # 1/c 1e-10
            return flam
        else:
            return None

    @staticmethod
    def flam_to_fnu(wave, flam):
        # erg/s/cm^2/A to ergs/cm**2/s/Hz
        if wave is not None and flam is not None:
            fnu = flam * 3.336e-19 * wave**2            # 1/c 1e-10
            return fnu
        else:
            return None

    @staticmethod
    def fnu_to_abmag(fnu):
        # erg/s/cm^2/Hz to mag_AB
        if fnu is not None:
            mags = -2.5 * np.log10(fnu) - 48.60
            return mags
        else:
            return None

    @staticmethod
    def abmag_to_fnu(abmag):
        # mag_AB to erg/s/cm^2/Hz
        if abmag is not None:
            fnu = 10**(-0.4 * (abmag + 48.60))
            return fnu
        else:
            return None

    @staticmethod
    def flam_to_abmag(wave, flam):
        # erg/s/cm^2/A to mag_AB
        if wave is not None and flam is not None:
            fnu = Physics.flam_to_fnu(wave, flam)
            mags = Physics.fnu_to_abmag(fnu)
            return mags
        else:
            return None

    @staticmethod
    def air_to_vac(wave):
        """
        Implements the air to vacuum wavelength conversion described in eqn 65 of
        Griesen 2006
        """
        wlum = wave * 1e5
        return (1 + 1e-6 * (287.6155 + 1.62887 / wlum**2 + 0.01360 / wlum**4)) * wave

    @staticmethod
    def air_to_vac_deriv(wave):
        """
        Eqn 66
        """
        wlum = wave * 1e5
        return (1+1e-6*(287.6155 - 1.62877/wlum**2 - 0.04080/wlum**4))

    @staticmethod
    def vac_to_air(wave):
        fact = 1.0 + 2.735182e-4 + 131.4182 / wave**2 + 2.76249e8 / wave**4
        fact = fact * (wave >= 2000) + 1.0 * (wave < 2000)
        return wave/fact

    @staticmethod
    def cm_to_pc(d):
        return d / 3.08567758128e19

    @staticmethod
    def pc_to_cm(d):
        return d * 3.08567758128e19

    @staticmethod
    def vel_to_z(vel):
        return vel * 1e3 / Physics.c

    @staticmethod
    def z_to_vel(vel):
        return Physics.c * vel * 1e-3       # km/s
    
    @staticmethod
    def stellar_radius(log_L, log_T_eff):
        """
        Calculate the radius of a star from its luminosity and effective temperature
        from the Stefan-Boltzmann law
        """

        sb = 5.6704e-5                          # grams s^-3 kelvin^-4
        lsun = 3.839e33                         # erg/s 
        l = lsun * (10 ** log_L)                # luminosity from isochrone is in log(L/lsun)
        t = 10 ** log_T_eff                     # T_eff from isochrone is in log(teff)
        radius = np.sqrt(l / (4 * np.pi * sb * t**4))
        return radius                           # radius in cm

    @staticmethod
    def synth_flux_from_flam(wave, flux, thru):
        """
        Calculate the synthetic flux in a filter from flux per unit wavelength.

        This is the flux that goes into the AB magnitude calculation.
        """

        num = trapezoid(flux * thru * wave, wave)
        den = trapezoid(thru / wave, wave)
        flux = num / den / Physics.c * 1e-10    # erg/s/cm^2/Hz
        return flux

    @staticmethod
    def synth_flux_from_fnu(wave, flux, thru):
        raise NotImplementedError()
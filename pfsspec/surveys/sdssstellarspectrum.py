from pfsspec.obsmod.stellarspectrum import StellarSpectrum
from pfsspec.surveys.sdssspectrum import SdssSpectrum

class SdssStellarSpectrum(StellarSpectrum, SdssSpectrum):
    def __init__(self, orig=None):
        StellarSpectrum.__init__(self, orig=orig)
        SdssSpectrum.__init__(self, orig=orig)

        if isinstance(orig, SdssStellarSpectrum):
            'ra', 'dec', 'redshift', 'redshift_err', 'snr', 'Fe_H', 'Fe_H_err', 'T_eff', 'T_eff_err', 'log_g', 'log_g_err', 'a_Fe', 'a_Fe_err', 'mag', 'mag_err', 'mag_u', 'mag_u_err', 'mag_g', 'mag_g_err', 'mag_r', 'mag_r_err', 'mag_i', 'mag_i_err', 'mag_z', 'mag_z_err', 'ext'
        else:
            pass

    def get_param_names(self):
        params = SdssSpectrum.get_param_names(self)
        params += StellarSpectrum.get_param_names(self)
        return params

    def print_info(self):
        SdssSpectrum.print_info(self)
        StellarSpectrum.print_info(self)
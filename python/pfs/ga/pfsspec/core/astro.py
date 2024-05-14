import numpy as np
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
import astropy.units as u

class Astro():

    @staticmethod
    def datetime_to_mjd(value):
        """
        Converts a Python datetime object to Modified Julian Date (MJD).

        Args:
            value (datetime.datetime): The input datetime object.

        Returns:
            float: The corresponding MJD.
        """
        t = Time(value)
        return t.mjd

    @staticmethod
    def mjd_to_datetime(value):
        """
        Converts Modified Julian Date (MJD) to a Python datetime object.

        Args:
            value (float): The input MJD.

        Returns:
            datetime.datetime: The corresponding datetime object.
        """
        t = Time(value, format='mjd')
        return t.datetime
    
    @staticmethod
    def __normalize_coord_time_obs(ra, dec, mjd, observatory=None):
        observatory = observatory if observatory is not None else 'Subaru'
        if isinstance(observatory, str):
            observatory = EarthLocation.of_site(observatory)

        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        time = Time(mjd, format='mjd')

        return coord, time, observatory
    
    @staticmethod
    def radec_to_altaz(ra, dec, mjd, observatory=None):
        """
        Converts RA, Dec to Alt, Az.

        Args:
            ra (float): The RA in degrees.
            deg (float): The Dec in degrees.
            mjd (float): The Modified Julian Date.
            observatory (str): The observatory name. Default is Subaru.

        Returns:
            tuple: The corresponding (Alt, Az) pair in degrees.
        """

        coord, time, observatory = Astro.__normalize_coord_time_obs(ra, dec, mjd, observatory)
        altaz = coord.transform_to(AltAz(obstime=time, location=observatory))

        return altaz.alt.deg, altaz.az.deg
            
    @staticmethod
    def v_corr(kind, ra, dec, mjd, observatory=None):
        # Calculate the barycentric or heliocentric velocity correction
        # Note that the barycentric correction is calculated using the optical
        # convention as v_corr = z * c so corrections to already determined
        # velocities should be calculated as v = v_obs + v_corr + v_obs * v_corr / c

        coord, time, observatory = Astro.__normalize_coord_time_obs(ra, dec, mjd, observatory)
        v_corr = coord.radial_velocity_correction(kind, obstime=time, location=observatory)
        return v_corr.to(u.km / u.s).value

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
    def __normalize_coord_time_obs(ra, dec, mjd=None, time=None, location=None):
        location = location if location is not None else 'Subaru'
        if isinstance(location, str):
            location = EarthLocation.of_site(location)

        coord = SkyCoord(ra=ra, dec=dec, unit='deg')

        if mjd is not None:
            time = Time(mjd, format='mjd')

        return coord, time, location
    
    @staticmethod
    def radec_to_altaz(ra, dec, mjd=None, time=None, location=None):
        """
        Converts RA, Dec to Alt, Az.

        Args:
            ra (float): The RA in degrees.
            deg (float): The Dec in degrees.
            mjd (float): The Modified Julian Date.
            time (astropy.time.Time): The observation time. If provided, it overrides `mjd`.
            location (astropy.coordinates.EarthLocation): The observatory location. Default is Subaru.

        Returns:
            tuple: The corresponding (Alt, Az) pair in degrees.
        """

        coord, time, location = Astro.__normalize_coord_time_obs(ra, dec, mjd=mjd, time=time, location=location)
        altaz = coord.transform_to(AltAz(obstime=time, location=location))

        return altaz.alt.deg, altaz.az.deg
            
    @staticmethod
    def v_corr(kind, ra, dec, mjd=None, time=None, location=None):
        # Calculate the barycentric or heliocentric velocity correction
        # Note that the barycentric correction is calculated using the optical
        # convention as v_corr = z * c so corrections to already determined
        # velocities should be calculated as v = v_obs + v_corr + v_obs * v_corr / c

        coord, time, location = Astro.__normalize_coord_time_obs(ra, dec, mjd=mjd, time=time, location=location)
        v_corr = coord.radial_velocity_correction(kind, obstime=time, location=location)
        return v_corr.to(u.km / u.s).value

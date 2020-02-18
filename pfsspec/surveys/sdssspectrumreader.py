import os
import sys
import numpy as np
from astropy.io import fits
from SciServer import Authentication, CasJobs

from pfsspec.surveys.survey import Survey
from pfsspec.surveys.surveyspectrumreader import SurveySpectrumReader
from pfsspec.surveys.sdssspectrum import SdssSpectrum

class SdssSpectrumReader(SurveySpectrumReader):
    def __init__(self):
        super(SdssSpectrumReader, self).__init__()
        self.path = None

    def read(self, file):
        self.redshift = 0.0

        loglambda0 = file[0].header['COEFF0  ']
        loglambda1 = file[0].header['COEFF1  ']
        numbins = file[0].data.shape[1]
        logwave = loglambda0 + loglambda1 * np.arange(0, numbins)

        spec = SdssSpectrum()
        spec.wave = wave = 10 ** logwave
        spec.flux = file[0].data[0, :]
        spec.flux_err = file[0].data[2, :]
        spec.flux_sky = file[0].data[4, :]
        spec.mask = np.int32(file[0].data[3, :])
        return spec

    def get_filename(mjd, plate, fiber, das='das2', ver='1d_26'):
        # .../das2/spectro/1d_26/0288/1d/spSpec-52000-0288-005.fit
        return '{:s}/spectro/{:s}/{:04d}/1d/spSpec-{:5d}-{:04d}-{:03d}.fit'.format(das, ver, int(plate), int(mjd), int(plate), int(fiber))

    def authenticate(self, username, password):
        self.sciserver_token = Authentication.login(username, password)

    def execute_query(self, sql, context='DR7'):
        return CasJobs.executeQuery(sql=sql, context=context, format="pandas")

    def find_stars(self, top=None, mjd=None, plate=None, Fe_H=None, T_eff=None, log_g=None, a_Fe=None):
        where = ''
        if mjd is not None:
            where += "AND s.mjd = {:d} \n".format(mjd)
        if plate is not None:
            where += "AND s.plate = {:d} \n".format(plate)
        if Fe_H is not None:
            where += "AND spp.feha BETWEEN {:f} AND {:f} \n".format(Fe_H[0], Fe_H[1])
        if T_eff is not None:
            where += "AND spp.teffa BETWEEN {:f} AND {:f} \n".format(T_eff[0], T_eff[1])
        if log_g is not None:
            where += "AND spp.logga BETWEEN {:f} AND {:f} \n".format(log_g[0], log_g[1])
        if a_Fe is not None:
            where += "AND spp.alphafe BETWEEN {:f} AND {:f} \n".format(a_Fe[0], a_Fe[1])

        sql = \
        """
        SELECT {} 
            s.specObjID AS id, s.mjd, s.plate, s.fiberID, s.ra AS ra, s.dec AS dec, 
            s.z AS redshift, s.zErr AS redshift_err, s.sn_1 AS snr,
            spp.feha AS Fe_H, spp.fehaerr AS Fe_H_err, 
            spp.teffa AS T_eff, spp.teffaerr AS T_eff_err,
            spp.logga AS log_g, spp.loggaerr AS log_g_err,
            spp.alphafe AS a_Fe, spp.alphafeerr AS a_Fe_err,
            p.psfMag_r AS mag, p.psfMagErr_r AS mag_err,
            p.psfMag_u AS mag_u, p.psfMagErr_u AS mag_u_err,
            p.psfMag_g AS mag_g, p.psfMagErr_g AS mag_g_err,
            p.psfMag_r AS mag_r, p.psfMagErr_r AS mag_r_err,
            p.psfMag_i AS mag_i, p.psfMagErr_i AS mag_i_err,
            p.psfMag_z AS mag_z, p.psfMagErr_z AS mag_z_err
        FROM SpecObj s
            INNER JOIN sppParams spp ON spp.specobjID = s.specObjID
            INNER JOIN PhotoObj p ON p.objID = s.bestObjID
        WHERE specClass = 1 AND zConf > 0.98 
              AND p.psfMag_r > 10               -- exclude unmeasured psf mag 
              {}
        ORDER BY s.mjd, s.plate, s.fiberID
        """.format('' if top is None else 'TOP {:d}'.format(top), where)

        return self.execute_query(sql)

    def load_spectrum(self, row):
        filename = SdssSpectrumReader.get_filename(row['mjd'], row['plate'], row['fiberID'])
        filename = os.path.join(self.path, filename)
        with fits.open(filename, memmap=False) as hdus:
            spec = self.read(hdus)
            return spec

import os
import logging
import getpass

from SciServer import Authentication, CasJobs

from pfsspec.data.surveyreader import SurveyReader

class SdssSurveyReader(SurveyReader):
    def __init__(self, orig=None):
        super(SdssSurveyReader, self).__init__(orig=orig)

        if isinstance(orig, SdssSurveyReader):
            self.user = orig.user
            self.token = orig.token
            self.reader = orig.reader

            self.dr = orig.dr

            self.plate = orig.plate
            self.mjd = orig.mjd
            # TODO: add more filters
        else:
            self.user = None
            self.token = None
            self.reader = None

            self.dr = None

            self.plate = None
            self.mjd = None
            # TODO: add more filters

    def add_args(self, parser):
        super(SdssSurveyReader, self).add_args(parser)

        parser.add_argument('--user', type=str, help='SciServer username\n')
        parser.add_argument('--token', type=str, help='SciServer auth token\n')

        parser.add_argument('--dr', type=str, default='DR7', choices=['DR7', 'DR16'], help='Data release')

        parser.add_argument('--plate', type=int, default=None, help='Limit to a single plate')
        parser.add_argument('--mjd', type=int, default=None, help='Limit to a single MJD')
        # TODO: add more filters

    def init_from_args(self, args):
        super(SdssSurveyReader, self).init_from_args(args)

        self.user = self.get_arg('user', self.user, args)
        self.token = self.get_arg('token', self.token, args)

        self.dr = self.get_arg('dr', self.dr, args)

        self.plate = self.get_arg('plate', self.plate, args)
        self.mjd = self.get_arg('mjd', self.mjd, args)
        # TODO: add more filters

    def create_auth_token(self):
        if self.token is None:
            if self.user is None:
                self.user = input('SciServer username: ')
            password = getpass.getpass()
            self.token = Authentication.login(self.user, password)
        self.logger.info('SciServer token: {}'.format(self.token))

    def authenticate(self, username, password):
        self.sciserver_token = Authentication.login(username, password)

    def execute_query(self, sql, context='DR7'):
        return CasJobs.executeQuery(sql=sql, context=context, format="pandas")

    def open_data(self, indir, outdir):
        super(SdssSurveyReader, self).open_data(indir, outdir)

        self.create_auth_token()

        self.reader = self.create_spectrum_reader()
        self.reader.path = indir
        self.reader.sciserver_token = self.token

    def find_objects(self):
        # TODO: this could be generic if we had a few more filters
        raise NotImplementedError()

    def run(self):
        self.logger.info('Querying SkyServer for spectrum headers')
        params = self.find_objects()
        self.logger.info('Found {} objects.'.format(params.shape[0]))

        self.logger.info('Start loading spectra from `{}`'.format(self.reader.path))
        self.load_survey(params)

    def execute_notebooks(self, script):
        super(SdssSurveyReader, self).execute_notebooks(script)

        script.execute_notebook('eval_import_sdss', parameters={
            'PFSSPEC_ROOT': os.environ['PFSSPEC_ROOT'],
            'PFSSPEC_DATA': os.environ['PFSSPEC_DATA'],
            'DATASET_PATH': self.outdir,
        })
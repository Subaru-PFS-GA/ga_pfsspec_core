import os
import glob

from ..setup_logger import logger
from pfs.ga.pfsspec.scripts.import_ import Import
from pfs.ga.pfsspec.stellarmod.boszspectrumreader import BoszSpectrumReader
from pfs.ga.pfsspec.stellarmod.boszatmreader import BoszAtmReader
from pfs.ga.pfsspec.stellarmod.boszatmgridreader import BoszAtmGridReader
from pfs.ga.pfsspec.stellarmod.boszatmgrid import BoszAtmGrid

class ImportBoszAtm(Import):
    # TODO: Convert into an import module
    def __init__(self):
        super(ImportBoszAtm, self).__init__()

    def add_args(self, parser, config):
        super(ImportBoszAtm, self).add_args(parser, config)
        parser.add_argument("--max", type=int, default=None, help="Stop after this many items.\n")
        parser.add_argument('--preload-arrays', action='store_true', help='Do not preload flux arrays to save memory\n')

    def run(self):
        super(ImportBoszAtm, self).run()

        grid = BoszAtmGrid()
        r = BoszAtmReader(self.args['path'])
        gr = BoszAtmGridReader(grid, r, self.args['max'])
        gr.parallel = self.threads != 1

        if 'preload_arrays' in self.args and self.args['preload_arrays'] is not None:
            grid.preload_arrays = self.args['preload_arrays']

        if os.path.isdir(self.args['path']):
            logger.info('Running in grid mode')

            # Load the first spectrum to get wavelength grid
            # fn = BoszSpectrumReader.get_filename(Fe_H=0.0, T_eff=5000.0, log_g=1.0, O_M=0.0, C_M=0.0, R=res)
            # fn = os.path.join(self.args['path'], fn)
            # spec = r.read(fn)
            raise NotImplementedError()
        else:
            logger.info('Running in file list mode')
            files = glob.glob(os.path.expandvars(self.args['path']))
            logger.info('Found {} files.'.format(len(files)))

        grid.init_values()
        grid.build_axis_indexes()
        grid.save(os.path.join(self.args['out'], 'atm.h5'), 'h5')

        if os.path.isdir(self.args['path']):
            raise NotImplementedError()
            #gr.read_grid()
        else:
            gr.read_files(files)

        #r.grid.build_flux_index(rebuild=True)
        grid.save(os.path.join(self.args['out'], 'atm.h5'), 'h5')

def main():
    script = ImportBoszAtm()
    script.execute()

if __name__ == "__main__":
    main()
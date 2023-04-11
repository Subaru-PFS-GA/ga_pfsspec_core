import os

from pfs.ga.pfsspec.scripts.import_ import Import
from pfs.ga.pfsspec.stellarmod.kuruczgrid import KuruczGrid
from pfs.ga.pfsspec.stellarmod.kuruczgridreader import KuruczGridReader
from pfs.ga.pfsspec.stellarmod.kuruczspectrumreader import KuruczSpectrumReader

class ImportKurucz(Import):
    # TODO: Convert into an import module
    def __init__(self):
        super(ImportKurucz, self).__init__()

    def add_args(self, parser, config):
        super(ImportKurucz, self).add_args(parser, config)
        parser.add_argument("--grid", type=str,
                                 choices=['kurucz', 'nover', 'anover', 'odfnew', 'aodfnew'],
                                 default='kurucz', help="Model subtype\n")

    def run(self):
        super(ImportKurucz, self).run()

        grid = KuruczGrid(self.args['grid'])
        reader = KuruczGridReader(grid)
        reader.read_grid(self.args['path'])
        grid.save(os.path.join(self.args['out'], 'spectra.npz'))

def main():
    script = ImportKurucz()
    script.execute()

if __name__ == "__main__":
    main()
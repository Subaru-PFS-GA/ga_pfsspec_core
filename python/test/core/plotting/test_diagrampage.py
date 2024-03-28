import os
import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.plotting import DiagramPage, Diagram

class DiagramPageTest(TestBase):

    def test_init_11(self):
        f = DiagramPage(1, 1)
        f.add_diagram((0, 0), Diagram())
        f.save(os.path.expandvars(os.path.join('{$PFSSPEC_TEST}', self.get_test_filename(ext='.pdf'))))

    def test_init_12(self):
        f = DiagramPage(1, 2)
        f.add_diagram((0, 0), Diagram())
        f.add_diagram((0, 1), Diagram())
        f.save(os.path.expandvars(os.path.join('{$PFSSPEC_TEST}', self.get_test_filename(ext='.pdf'))))

    def test_init_21(self):
        f = DiagramPage(2, 1)
        f.add_diagram((0, 0), Diagram())
        f.add_diagram((1, 0), Diagram())
        f.save(os.path.expandvars(os.path.join('{$PFSSPEC_TEST}', self.get_test_filename(ext='.pdf'))))

    def test_init_22(self):
        f = DiagramPage(2, 2)
        f.add_diagram((0, 0), Diagram())
        f.add_diagram((1, 0), Diagram())
        f.add_diagram((0, 1), Diagram())
        f.add_diagram((1, 1), Diagram())
        f.save(os.path.expandvars(os.path.join('{$PFSSPEC_TEST}', self.get_test_filename(ext='.pdf'))))

    def test_init_multipage(self):
        f = DiagramPage(2, 1, 1)
        f.add_diagram((0, 0, 0), Diagram())
        f.add_diagram((1, 0, 0), Diagram())
        f.save(os.path.expandvars(os.path.join('{$PFSSPEC_TEST}', self.get_test_filename(ext='.pdf'))))
        f.save(os.path.expandvars(os.path.join('{$PFSSPEC_TEST}', self.get_test_filename(ext='.png'))))
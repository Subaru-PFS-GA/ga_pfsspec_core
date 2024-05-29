import os
from datetime import datetime

from pfs.ga.pfsspec.core.util.copy import *
from .entry import Entry

class History():
    def __init__(self, orig=None):
        self.reset()

        if not isinstance(orig, History):
            pass
        else:
            self.entries = safe_deep_copy(orig.entries)

    def copy(self):
        return History(orig=self)

    def reset(self):
        self.entries = []

    def append(self, message, data=None):
        entry = Entry(datetime.now, message, data=data)
        self.entries.append(entry)

    def save(self, filename):
        with open(filename, "w") as f:
            for e in self.entries:
                f.write(e.message + '\n')
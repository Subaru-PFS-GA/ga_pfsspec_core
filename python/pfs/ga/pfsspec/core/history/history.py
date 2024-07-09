import os
from datetime import datetime
from io import StringIO

from pfs.ga.pfsspec.core.util.copy import *
from .entry import Entry

class History():
    def __init__(self, orig=None):
        self.reset()

        if not isinstance(orig, History):
            pass
        else:
            self.entries = safe_deep_copy(orig.entries)

    def __str__(self) -> str:
        # Write the history to a string using a string writer.
        with StringIO() as f:
            self.save(f)
            return f.getvalue()

    def copy(self):
        return History(orig=self)

    def reset(self):
        self.entries = []

    def append(self, message, data=None):
        entry = Entry(datetime.now, message, data=data)
        self.entries.append(entry)

    def save(self, file):

        def write_entries(f):
            for e in self.entries:
                f.write(e.message + '\n')

        if isinstance(file, str):
            with open(file, "a") as f:
                write_entries(f)
        else:
            write_entries(file)

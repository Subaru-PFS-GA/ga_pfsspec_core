import numpy as np

class GridEnumerator():
    def __init__(self, grid=None, s=None, top=None, resume=False):
        self.grid = grid
        self.slice = s
        self.i = 0
        self.top = top
        self.stop = False
        self.resume = resume

        self.limits = None
        self.current = None

    def __iter__(self):
        self.limits = [ a.values.shape[0] for i, p, a in self.grid.enumerate_axes(s=self.slice) ]
        self.current = [ 0 for i, p, a in self.grid.enumerate_axes(s=self.slice) ]
        return self

    def __len__(self):
        if self.resume:
            mask = None
            for name in self.grid.value_indexes:
                m = self.grid.value_indexes[name]
                if mask is None:
                    mask = m
                else:
                    mask |= m
            s = np.sum(~mask)
        else:
            s = 1
            for i, p, axis in self.grid.enumerate_axes(s=self.slice):
                s *= axis.values.shape[0]

        if self.top is not None:
            return min(s, self.top)
        else:
            return s

    def __next__(self):
        if self.stop:
            raise StopIteration()
        else:
            # If in continue mode, we need to skip items that are already in the grid
            while True:
                ci = self.current.copy()
                cr = { p: axis.values[self.current[i]] for i, p, axis in self.grid.enumerate_axes(s=self.slice) }

                k = len(self.limits) - 1
                while k >= 0:
                    self.current[k] += 1
                    if self.current[k] == self.limits[k]:
                        self.current[k] = 0
                        k -= 1
                        continue
                    else:
                        break

                if k == -1 or self.i == self.top:
                    self.stop = True

                self.i += 1

                if self.resume:
                    # Test if item is already in the grid
                    mask = None
                    for name in self.grid.value_indexes:
                        m = self.grid.has_value_at(name, ci)
                        if mask is None:
                            mask = m
                        else:
                            mask |= m

                    if not mask:
                        # Item does not exist
                        break
                else:
                    break

            return ci, cr
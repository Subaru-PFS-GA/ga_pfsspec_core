import numpy as np

class GridConfig():
    """
    Implements functions to initialize a model grid. Inherited classes should
    implement grid-specific functionality in overridden functions.
    """

    def __init__(self, orig=None):
        if isinstance(orig, GridConfig):
            pass
        else:
            pass

    def init_axes(self, grid):
        """
        When implemented in derived classes, initialzes the axes of the grid by
        calling `Grid.init_axis` for each axis.

        :param grid: The grid to be initialized.
        """
        raise NotImplementedError()

    def init_values(self, grid):
        """
        When implemented in derived classes, inizializes the value arrays of the
        grid by calling `Grid.init_value` for each value array.

        :param grid: The grid to be initialized.
        """
        raise NotImplementedError()

    def allocate_values(self, grid):
        """
        When implemented in derived classes, allocate the value arrays of the
        grid by calling `Grid.allocate_value` for each value array.

        :param grid: The grid to be initialized.
        """
        raise NotImplementedError()

    def is_value_valid(self, grid, name, value):
        """
        Returns True when `value` is to be treated as valid. This function is
        called to determine if a particular data array is valid or not to be
        marked in the corresponding index array.

        :param grid: Reference to the grid
        :param name: Name of the value array to be checked for validity.
        :param value: The value of the array to be checked for validity.
        """

        return np.logical_not(np.any(np.isnan(value), axis=-1))

    def get_chunk_shape(self, grid, name, shape, s=None):
        """
        When implemented in derived classes, it returns a chunk size for a
        particular value array identified by name.

        :param grid: Reference to the grid.
        :param name: Name of the value array.
        :param shape: Shape of the value array.
        :param s: Optional slicing (reserved, not used)
        """
        
        return None

class DatasetConfig():
    """
    Implements functions to initialize a Dataset by configuring value arrays.
    """

    def __init__(self, orig=None):
        if isinstance(orig, DatasetConfig):
            pass
        else:
            pass

    def init_values(self, dataset):
        pass

    def allocate_values(self, dataset):
        pass

    def get_chunk_shape(self, dataset, name, shape, s=None):
        return None
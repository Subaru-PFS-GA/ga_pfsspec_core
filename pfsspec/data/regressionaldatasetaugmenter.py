import numpy as np

from pfsspec.data.datasetaugmenter import DatasetAugmenter

class RegressionalDatasetAugmenter(DatasetAugmenter):
    def __init__(self, orig=None):
        super(RegressionalDatasetAugmenter, self).__init__(orig=orig)

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, partitions=None, batch_size=None, shuffle=None, chunk_size=None, seed=None):
        input_shape = dataset.shape
        output_shape = (len(labels),)
        d = super(RegressionalDatasetAugmenter, cls).from_dataset(input_shape, output_shape,
                                                                  dataset, labels, coeffs,
                                                                  weight=weight,
                                                                  partitions=partitions,
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  chunk_size=chunk_size,
                                                                  seed=seed)
        return d

    def add_args(self, parser):
        super(RegressionalDatasetAugmenter, self).add_args(parser)

    def init_from_args(self, args):
        super(RegressionalDatasetAugmenter, self).init_from_args(args)

    def scale_output(self, output):
        values = output / self.coeffs
        return super(RegressionalDatasetAugmenter, self).scale_output(values)

    def rescale_output(self, output):
        values = output * self.coeffs
        return super(RegressionalDatasetAugmenter, self).rescale_output(values)

    def set_batch(self, batch_id, input, output):
        chunk_id, idx = self.get_batch_index(batch_id)
        self.dataset.set_params(self.labels, output, idx, self.chunk_size, chunk_id)

    def augment_batch(self, chunk_id, idx):
        input, output, weight = super(RegressionalDatasetAugmenter, self).augment_batch(chunk_id, idx)

        input = np.array(self.dataset.get_flux(idx, self.chunk_size, chunk_id), copy=True, dtype=np.float)
        output = np.array(self.dataset.get_params(self.labels, idx, self.chunk_size, chunk_id))

        return input, output, weight

    def get_output_labels(self, model):
        if model.mc_count is None:
            return self.labels, ['_pred']
        else:
            return self.labels, ['_pred', '_std', '_skew', '_median']

    def init_output_labels(self, labels, postfixes):
        self.labels = []
        for label in labels:                
            for postfix in postfixes:
                self.labels.append(label + postfix)
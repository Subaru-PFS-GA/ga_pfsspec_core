import numpy as np

from pfsspec.data.datasetaugmenter import DatasetAugmenter

class SdssAugmenter(DatasetAugmenter):
    def __init__(self):
        super(SdssAugmenter, self).__init__()

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, batch_size=1, shuffle=True, seed=0):
        input_shape = dataset.flux.shape
        output_shape = (len(labels),)
        d = super(SdssAugmenter, cls).from_dataset(dataset, labels, coeffs, weight,
                                            input_shape, output_shape,
                                            batch_size=batch_size, shuffle=shuffle, seed=seed)
        return d

    def augment_batch(self, batch_index):
        flux, labels, weight = super(SdssAugmenter, self).augment_batch(batch_index)

        if self.multiplicative_bias:
            bias = np.random.uniform(0.8, 1.2, (flux.shape[0], 1))
            flux = flux * bias
        if self.additive_bias:
            bias = np.random.normal(0, 1.0, (flux.shape[0], 1))
            flux = flux + bias

        if self.include_wave:
            nflux = np.zeros((len(batch_index), self.dataset.flux.shape[1], 2))
            nflux[:, :, 0] = flux
            nflux[:, :, 1] = self.dataset.wave
            flux = nflux

        return flux, labels, weight

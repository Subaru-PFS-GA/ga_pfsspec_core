import numpy as np

from pfsspec.data.datasetaugmenter import DatasetAugmenter

class SdssAugmenter(DatasetAugmenter):
    """Implements data augmentation to train on observed SDSS spectra."""

    def __init__(self, orig=None):
        super(SdssAugmenter, self).__init__(orig=orig)

    @classmethod
    def from_dataset(cls, dataset, labels, coeffs, weight=None, batch_size=1, shuffle=True, chunk_size=None, seed=0):
        # TODO: verify order of parameters
        raise NotImplementedError()
        
        input_shape = dataset.shape
        output_shape = (len(labels),)
        d = super(SdssAugmenter, cls).from_dataset(dataset, labels, coeffs, weight,
                                            input_shape, output_shape,
                                            batch_size=batch_size, shuffle=shuffle, 
                                            chunk_size=chunk_size, seed=seed)
        return d

    def augment_batch(self, chunk_id, idx):
        flux, labels, weight = super(SdssAugmenter, self).augment_batch(chunk_id, idx)

        if self.multiplicative_bias:
            bias = np.random.uniform(0.8, 1.2, (flux.shape[0], 1))
            flux = flux * bias
        if self.additive_bias:
            bias = np.random.normal(0, 1.0, (flux.shape[0], 1))
            flux = flux + bias

        if self.include_wave:
            nflux = np.zeros((len(idx), self.dataset.shape[1], 2))
            nflux[:, :, 0] = flux            
            nflux[:, :, 1] = self.dataset.wave
            flux = nflux

        return flux, labels, weight

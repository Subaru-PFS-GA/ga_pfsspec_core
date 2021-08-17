import numpy as np

import pfsspec.util as util
from pfsspec.common.spectrum import Spectrum
from pfsspec.ml.dnn.keras.kerasdataaugmenter import KerasDataAugmenter

class DatasetAugmenter(KerasDataAugmenter):
    # The issue to solve is that in order
    # to get a reasonable training performance, chunks of batches have to be
    # kept together and read from the storage in one single operation. Chunks
    # will be read in random order but they're large enough to support sustained
    # sequential reads.
    #
    # This class also implements very basic augmentation functionality such as
    # offsetting and scaling, as well as masking invalid and outlier data.
    # TODO: maybe these should be shoveled into an auxilliary class to focus on
    #       dataset chunking here.

    def __init__(self, orig=None):
        super(DatasetAugmenter, self).__init__(orig=orig)

        if isinstance(orig, DatasetAugmenter):
            self.dataset = orig.dataset
            
            self.chunk_count = orig.chunk_count
            self.chunk_size = orig.chunk_size
            self.chunk_index = orig.chunk_index

            self.labels = orig.labels
            self.coeffs = orig.coeffs
            self.weight = orig.weight

            self.noise = orig.noise
            self.noise_schedule = orig.noise_schedule

            self.aug_offset = orig.aug_offset
            self.aug_scale = orig.aug_scale
            self.mask_data = orig.mask_data
            self.mask_random = orig.mask_random
            self.mask_value = orig.mask_value
            self.lowsnr = orig.lowsnr
            self.lowsnr_value = orig.lowsnr_value
            self.extreme = orig.extreme
            self.extreme_value = orig.extreme_value

            self.include_wave = orig.include_wave
        else:
            self.dataset = None
            
            self.chunk_count = None
            self.chunk_size = None
            self.chunk_index = None

            self.labels = None
            self.coeffs = None
            self.weight = None

            self.noise = None
            self.noise_schedule = None

            self.aug_offset = None
            self.aug_scale = None
            self.mask_data = False
            self.mask_random = None
            self.mask_value = [0]
            self.lowsnr = None
            self.lowsnr_value = [0.0, 1.0]
            self.extreme = None
            self.extreme_value = [0.0, 1.0]

            self.include_wave = False

    @classmethod
    def from_dataset(cls, input_shape, output_shape, dataset, labels, coeffs, weight=None, partitions=None, batch_size=None, shuffle=None, chunk_size=None, seed=None):
        d = super(DatasetAugmenter, cls).from_shapes(input_shape, output_shape, partitions=partitions, batch_size=batch_size, shuffle=shuffle, seed=seed)

        d.dataset = dataset
        d.chunk_size = chunk_size
        d.labels = labels
        d.coeffs = coeffs
        d.weight = weight
               
        return d

    #region Command-line arguments

    def add_args(self, parser):
        super(DatasetAugmenter, self).add_args(parser)

        parser.add_argument('--noise', type=float, default=None, help='Add noise.\n')
        parser.add_argument('--noise-sch', type=str, choices=['constant', 'linear'], default='constant', help='Noise schedule.\n')

        parser.add_argument('--aug-offset', type=float, default=None, help='Augment by adding a random offset.\n')
        parser.add_argument('--aug-scale', type=float, default=None, help='Augment by multiplying with a random number.\n')
        parser.add_argument('--mask-data', action='store_true', dest='mask_data', help='Use mask from dataset.\n')
        parser.add_argument('--no-mask-data', action='store_false', dest='mask_data', help='Use mask from dataset.\n')
        parser.add_argument('--mask-random', type=float, nargs="*", help='Add random mask.\n')
        parser.add_argument('--mask-value', type=float, nargs="*", default=[0], help='Use mask value.\n')
        parser.add_argument('--lowsnr', type=float, help='Pixels that are considered low SNR.\n')
        parser.add_argument('--lowsnr-value', type=float, nargs="*", default=[0.0, 1.0], help='Randomize noisy bins that are below snr.\n')
        parser.add_argument('--extreme', type=float, help='Pixels that are considered extreme values.\n')
        parser.add_argument('--extreme-value', type=float, nargs="*", default=[0.0, 1.0], help='Randomize extreme value bins.\n')

        parser.add_argument('--include-wave', action='store_true', help='Include wave vector in training.\n')

    def init_from_args(self, args):
        super(DatasetAugmenter, self).init_from_args(args)

        self.noise = util.get_arg('noise', self.noise, args)
        if self.mode == 'train':
            self.noise_schedule = util.get_arg('noise_sch', self.noise_schedule, args)
        elif self.mode in ['test', 'predict']:
            self.noise_schedule = 'constant'
        else:
            raise NotImplementedError()

        self.aug_offset = util.get_arg('aug_offset', self.aug_offset, args)
        self.aug_scale = util.get_arg('aug_scale', self.aug_scale, args)

        self.mask_data = util.get_arg('mask_data', self.mask_data, args)
        self.mask_random = util.get_arg('mask_random', self.mask_random, args)
        self.mask_value = util.get_arg('mask_value', self.mask_value, args)

        self.lowsnr = util.get_arg('lowsnr', self.lowsnr, args)
        self.lowsnr_value = util.get_arg('lowsnr_value', self.lowsnr_value, args)
        self.extreme = util.get_arg('extreme', self.extreme, args)
        self.extreme_value = util.get_arg('extreme_value', self.extreme_value, args)

        self.include_wave = self.get_arg('include_wave', self.include_wave, args)

    #endregion

    def get_chunk_count(self):
        return self.chunk_count

    def reshuffle(self):
        if self.chunk_size is None:
            super(DatasetAugmenter, self).reshuffle()
        else:
            # Shuffle chunks and the batches inside chunks. By keeping the
            # chunk size big enough, sustained sequential reads from the storage
            # will be possible. 

            # First we shuffle chunks then shuffle batch_ids within chunks only.
            # We assume that batch_ids can be converted into chunk_ids by simple
            # integer division.
            # Next, data items are shuffled within chunks.
            # When using chunks, data index should be relative to the chunk.

            if self.chunk_size is not None and self.chunk_size % self.batch_size != 0:
                raise Exception('Chunk size must be a multiple of batch size.')

            if self.chunk_size is not None and self.filter is not None:
                raise Exception('Data augmenter filtering and data chunking cannot be used together.')

            if self.rng is None:
                self.rng = np.random.RandomState(self.seed)

            self.input_count = self.dataset.shape[0]
            self.chunk_count = np.int32(np.ceil(self.input_count / self.chunk_size))
            self.batch_count = np.int32(np.ceil(self.input_count / self.batch_size))

            # Override partition count if chunk_count is too small
            if self.partitions > self.chunk_count:
                raise Exception('Training set is too small, please reduce thread count from {}.'.format(self.partitions))
            
            # Shuffle chunks
            ci = np.arange(self.chunk_count)
            if self.shuffle:
                ci = ci[self.rng.permutation(ci.shape[0])]

            # Split up chunks among worker threads
            self.chunk_index = np.array_split(ci, self.partitions)

            # Split up batches among worker threads but make sure they're
            # kept together within a chunk
            self.batch_index = []
            for s in range(self.partitions):
                bb = []
                for chunk_id in self.chunk_index[s]:
                    batch_start = chunk_id * (self.chunk_size // self.batch_size)
                    batch_end = min(self.batch_count, (chunk_id + 1) * (self.chunk_size // self.batch_size))
                    bi = np.arange(batch_start, batch_end)
                    if self.shuffle:
                        bi = bi[self.rng.permutation(bi.shape[0])]
                    bb.append(bi)
                self.batch_index.append(np.concatenate(bb))

            # Shuffle data items within chunks, use relative indexing
            di = []
            for chunk_id in range(self.chunk_count):
                idx_start = chunk_id * self.chunk_size
                idx_end = min(self.input_count, (chunk_id + 1) * self.chunk_size)
                idx = np.arange(idx_end - idx_start)
                if self.shuffle:
                    idx = idx[self.rng.permutation(idx.shape[0])]
                di.append(idx)
            self.data_index = np.concatenate(di)

    def get_batch_index(self, batch_id):
        _, batch_idx = super(DatasetAugmenter, self).get_batch_index(batch_id)

        # Calculate the chunk_id from batch_id if chunking is turned on
        if self.chunk_size is None:
            chunk_id = None
        else:
            chunk_id = batch_id // (self.chunk_size // self.batch_size)
        
        return chunk_id, batch_idx

    def get_batch(self, batch_id):
        chunk_id, idx = self.get_batch_index(batch_id)
        input, output, weight = self.augment_batch(chunk_id, idx)
        return input, output, weight

    def augment_batch(self, chunk_id, idx):
        input = None
        output = None

        if self.weight is not None and 'weight' in self.dataset.params.columns:
            weight = np.array(self.dataset.get_params(['weight'], idx, self.chunk_size, chunk_id), copy=True, dtype=np.float)[..., 0]
        else:
            weight = None

        return input, output, weight

    def get_output_labels(self, model):
        # Override this to return list of labels and postfixes for prediction
        pass

    def init_output_labels(self, labels, postfixes):
        # Override this if new labels need to be created for prediction
        pass






    # TODO: these should go to a SpectrumAugmenter auxilliary class

    def noise_scheduler_linear_onestep(self):
        break_point = int(0.5 * self.total_epochs)
        if self.current_epoch < break_point:
            return self.current_epoch / break_point
        else:
            return 1.0

    def noise_scheduler_linear_twostep(self):
        break_point_1 = int(0.2 * self.total_epochs)
        break_point_2 = int(0.5 * self.total_epochs)
        if self.current_epoch < break_point_1:
            return 0.0
        elif self.current_epoch < break_point_2:
            return (self.current_epoch - break_point_1) / (self.total_epochs - break_point_1 - break_point_2)
        else:
            return 1.0

    def get_data_mask(self, chunk_id, idx, flux, mask):
        # Take mask from dataset
        if self.mask_data and self.dataset.has_mask():
            # TODO: verify this with real survey data
            mask = mask | (self.dataset.get_mask(idx, self.chunk_size, chunk_id) != 0)

        return mask

    def generate_random_mask(self, chunk_id, idx, flux, mask):
        # Generate random mask. First pick the number of masked intervals, then
        # generate the central wavelength of each and the normally distributed length.
        if self.mask_random is not None and self.mask_value is not None:
            # TODO: what if wave is not a constant vector?
            wl = self.dataset.get_wave(idx, self.chunk_size, chunk_id)
            max_n = int(self.mask_random[0])

            # Generate parameters of random intervals
            n = np.random.randint(0, max_n + 1, size=(flux.shape[0],))
            wc = wl[0] + np.random.rand(flux.shape[0], max_n) * (wl[-1] - wl[0])
            ww = np.maximum(0.0, np.random.normal(self.mask_random[1], self.mask_random[2], size=(flux.shape[0], max_n)))

            # Find corresponding indices
            mx = np.digitize([wc - ww / 2, wc + ww / 2], wl)  # shape: (2, flux.shape[0], max_n)

            # TODO: Can we vectorize this? The number of masks varies from spectrum to spectrum.
            for k in range(flux.shape[0]):
                for l in range(n[k]):
                    mask[k, mx[0, k, l]:mx[1, k, l]] = True

        return mask

    def get_error(self, chunk_id, idx):
        if self.dataset.has_error():
            return self.dataset.get_error(idx, self.chunk_size, chunk_id)
        else:
            return None

    def generate_noise(self, chunk_id, idx, flux):
        error = self.get_error(chunk_id, idx)
        noise = self.noise

        if self.noise_schedule == 'constant':
            pass
        elif self.noise_schedule == 'linear':
            noise *= self.noise_scheduler_linear_onestep()
        elif self.noise_schedule is not None:
            raise NotImplementedError()

        if noise is not None and noise > 0.0:
            # Noise don't have to be reproducible during training, do not reseed.
            flux = Spectrum.generate_noise(flux, noise, error=error, random_seed=None)

        return flux, error

    def augment_flux(self, chunk_id, idx, flux):
        # Additive and multiplicative bias, two numbers per spectrum
        if self.mode in ['train', 'test']:
            if self.aug_scale is not None:
                bias = np.random.normal(1, self.aug_scale, (flux.shape[0], 1))
                flux *= bias
            if self.aug_offset is not None:
                bias = np.random.normal(0, self.aug_offset, (flux.shape[0], 1))
                flux += bias

        return flux

    def apply_mask(self, flux, error, mask):
        # Set masked pixels to mask_value
        if self.mask_value is not None and mask is not None:
            if len(self.mask_value) == 0:
                flux[mask] = self.mask_value[0]
            else:
                flux[mask] = np.random.uniform(*self.mask_value, size=np.sum(mask))

        return flux

    def cut_lowsnr(self, flux, error):
        # Mask out points where noise is too high
        if self.lowsnr is not None and self.lowsnr_value is not None and error is not None:
            mask = (error == 0.0) 
            mask[~mask] |= (np.abs(flux[~mask] / error[~mask]) < self.lowsnr)
            if len(self.lowsnr_value) == 1:
                flux[mask] = self.lowsnr_value[0]
            else:
                flux[mask] = np.random.uniform(*self.lowsnr_value, size=np.sum(mask))

        return flux

    def cut_extreme(self, flux, error):
        # Mask out extreme values of flux
        mask = None
        if self.extreme is not None:
            if len(self.extreme) == 2:
                mask = (flux < self.extreme[0]) | (self.extreme[1] < flux)
            elif len(self.extreme) == 1:
                mask = (np.abs(flux) > self.extreme[0])
            else:
                raise NotImplementedError()

        if self.extreme_value is not None and mask is not None:
            if len(self.extreme_value) == 1:
                flux[mask] = self.extreme_value[0]
            else:
                flux[mask] = np.random.uniform(*self.extreme_value, size=np.sum(mask))

        return flux

    # def include_wave(self, flux, error, labels, weight, mask):
    #     if self.include_wave:
    #         nflux = np.zeros((len(idx), self.dataset.shape[1], 2))
    #         nflux[:, :, 0] = flux            
    #         nflux[:, :, 1] = self.dataset.wave
    #         flux = nflux

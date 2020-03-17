import os
import logging
import numpy as np

import pysynphot, pysynphot.binning, pysynphot.spectrum, pysynphot.reddening

import pfsspec.util as util
from pfsspec.obsmod.spectrum import Spectrum

class KuruczAugmenter():
    def __init__(self):
        self.noise = None
        self.noise_schedule = None
        self.aug_offset = None
        self.aug_scale = None
        self.mask_data = False
        self.mask_random = None
        self.mask_value = [0]
        self.lowsnr = None
        self.lowsnr_value = [0.0, 1.0]

        self.calib_bias = None
        self.calib_bias_count = None
        self.calib_bias_bandwidth = 200
        self.calib_bias_amplitude = 0.01

        self.ext = None
        self.ext_count = None
        self.ext_dist = None

    def add_args(self, parser):
        parser.add_argument('--noise', type=float, default=None, help='Add noise.\n')
        parser.add_argument('--noise-sch', type=str, choices=['constant', 'linear'], default='constant', help='Noise schedule.\n')
        parser.add_argument('--aug-offset', type=float, default=None, help='Augment by adding a random offset.\n')
        parser.add_argument('--aug-scale', type=float, default=None, help='Augment by multiplying with a random number.\n')
        parser.add_argument('--mask-data', action='store_true', help='Use mask from dataset.\n')
        parser.add_argument('--mask-random', type=float, nargs="*", help='Add random mask.\n')
        parser.add_argument('--mask-value', type=float, nargs="*", default=[0], help='Use mask value.\n')
        parser.add_argument('--lowsnr', type=float, help='Pixels that are considered low SND.\n')
        parser.add_argument('--lowsnr-value', type=float, nargs="*", default=[0.0, 1.0], help='Randomize noisy bins that are below snr.\n')
        
        parser.add_argument('--calib-bias', type=float, nargs=3, default=None, help='Add simulated calibration bias.')
        
        parser.add_argument('--ext', type=float, nargs='*', default=None, help='Extinction or distribution parameters.\n')

    def init_from_args(self, args):
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

        if 'calib_bias' in args and args['calib_bias'] is not None:
            calib_bias = args['calib_bias']
            self.calib_bias_count = int(calib_bias[0])
            self.calib_bias_bandwidth = calib_bias[1]
            self.calib_bias_amplitude = calib_bias[2]

        if 'ext' in args and args['ext'] is not None:
            ext = args['ext']
            self.ext_count = int(ext[0])
            self.ext_dist = [ext[1], ext[2]]

    def on_epoch_end(self):
        self.calib_bias = None
        self.ext = None

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

    def augment_batch(self, dataset, idx, flux, labels, weight):
        mask = np.full(flux.shape, False)
        mask = self.get_data_mask(dataset, idx, flux, labels, weight, mask)
        mask = self.generate_random_mask(dataset, idx, flux, labels, weight, mask)
        
        flux = self.apply_ext(dataset, idx, flux, labels, weight)
        flux = self.apply_calib_bias(dataset, idx, flux, labels, weight)
        flux, error = self.generate_noise(dataset, idx, flux, labels, weight)
        flux = self.augment_flux(dataset, idx, flux, labels, weight)

        flux = self.apply_lowsnr(dataset, idx, flux, error, labels, weight)
        flux = self.apply_mask(dataset, idx, flux, error, labels, weight, mask)

        return flux, labels, weight

    def get_data_mask(self, dataset, idx, flux, labels, weight, mask):       
        # Take mask from dataset
        if self.mask_data and dataset.mask is not None:
            # TODO: verify this with real survey data
            mask = mask | (dataset.mask[idx] != 0)

        return mask

    def generate_random_mask(self, dataset, idx, flux, labels, weight, mask):
        # Generate random mask
        if self.mask_random is not None and self.mask_value is not None:
            for k in range(flux.shape[0]):
                n = np.random.randint(0, self.mask_random[0] + 1)
                for i in range(n):
                    wl = dataset.wave[0] + np.random.rand() * (dataset.wave[-1] - dataset.wave[0])
                    ww = max(0.0, np.random.normal(self.mask_random[1], self.mask_random[2]))
                    mx = np.digitize([wl - ww / 2, wl + ww / 2], dataset.wave)
                    mask[k, mx[0]:mx[1]] = True

        return mask

    def generate_noise(self, dataset, idx, flux, labels, weight):
        if dataset.error is not None:
            error = dataset.error[idx]
        else:
            error = None

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

    def augment_flux(self, dataset, idx, flux, labels, weight):
        # Additive and multiplicative bias, two numbers per spectrum
        if self.aug_scale is not None:
            bias = np.random.normal(1, self.aug_scale, (flux.shape[0], 1))
            flux *= bias
        if self.aug_offset is not None:
            bias = np.random.normal(0, self.aug_offset, (flux.shape[0], 1))
            flux += bias

        return flux

    def apply_mask(self, dataset, idx, flux, error, labels, weight, mask):
        # Set masked pixels to mask_value
        if self.mask_value is not None and mask is not None:
            if len(self.mask_value) == 0:
                flux[mask] = self.mask_value[0]
            else:
                flux[mask] = np.random.uniform(*self.mask_value, size=np.sum(mask))

        return flux

    def apply_lowsnr(self, dataset, idx, flux, error, labels, weight):
        # Mask out points where noise is too high
        if self.lowsnr is not None and self.lowsnr_value is not None and error is not None:
            mask = (np.abs(flux / error) < self.lowsnr)
            if len(self.lowsnr_value) == 1:
                flux[mask] = self.lowsnr_value[0]
            else:
                flux[mask] = np.random.uniform(*self.lowsnr_value, size=np.sum(mask))

        return flux

    def apply_calib_bias(self, dataset, idx, flux, labels, weight):
        if self.calib_bias_count is not None:
            if self.calib_bias is None:
                self.generate_calib_bias(dataset.wave)
        
            # Pick calibration bias curve
            i = np.random.randint(self.calib_bias.shape[0], size=(flux.shape[0],))
            flux *= self.calib_bias[i, :]
        
        return flux

    def generate_calib_bias(self, wave):
        logging.info('Generating {} realization of calibration bias'.format(self.calib_bias_count))
        
        wave = wave if len(wave.shape) == 1 else wave[0]
        self.calib_bias = np.empty((self.calib_bias_count, wave.shape[0]))
        for i in range(self.calib_bias_count):
            self.calib_bias[i, :] = Spectrum.generate_calib_bias(wave,
                                                                 bandwidth=self.calib_bias_bandwidth,
                                                                 amplitude=self.calib_bias_amplitude)

        logging.info('Generated {} realization of calibration bias'.format(self.calib_bias_count))

    def apply_ext(self, dataset, idx, flux, labels, weight):
        if self.ext_count is not None:
            if self.ext is None:
                self.generate_ext(dataset.wave)

            # Pick extinction curve
            i = np.random.randint(self.ext.shape[0], size=(flux.shape[0],))
            flux *= self.ext[i, :]

        return flux

    def generate_ext(self, wave):
        logging.info('Generating {} realization of extinction curve'.format(self.ext_count))

        wave = wave if len(wave.shape) == 1 else wave[0]
        self.ext = np.empty((self.ext_count, wave.shape[0]))
        for i in range(self.ext_count):
            extval = np.random.uniform(*self.ext_dist)
            spec = pysynphot.spectrum.ArraySourceSpectrum(wave=wave, flux=np.full(wave.shape, 1.0), keepneg=True)
            obs = spec * pysynphot.reddening.Extinction(extval, 'mwavg')
            self.ext[i, :] = obs.flux

        logging.info('Generated {} realization of extinction curve'.format(self.ext_count))
import os
import numpy as np

from .datasetmerger import DatasetMerger
from .spectrumdataset import SpectrumDataset

class SpectrumDatasetMerger(DatasetMerger):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, SpectrumDatasetMerger):
            self.operation = 'sum'

            self.input_constant_wave = None
            self.input_wave_count = None
        else:
            self.operation = orig.operation

            self.input_constant_wave = orig.input_constant_wave
            self.input_wave_count = orig.input_wave_count

    def add_args(self, parser, config):
        super().add_args(parser, config)

        parser.add_argument('--operation', type=str, required=True, choices=['sum'], help='Operation to perform on the input datasets.\n')

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

    def create_dataset(self, preload_arrays=False):
        return SpectrumDataset(preload_arrays=preload_arrays)
    
    def open_input_data(self, input_paths):

        super().open_input_data(input_paths)

        # Verify that the wavelengths match up
        self.input_wave_count = None

        for ds in self.input_datasets:
            if self.input_wave_count is None:
                self.input_wave_count = ds.wave.shape[-1]
                self.input_constant_wave = ds.constant_wave
            elif self.input_wave_count != ds.wave.shape[-1]:
                raise Exception('Dataset wave counts are not equal.')
            elif self.input_constant_wave != ds.constant_wave:
                raise Exception('Some of the datasets have varying wave vectors.')

    def open_output_data(self, output_path):
        # Initialize output dataset
        super().open_output_data(output_path)

        # Allocate data arrays
        ds = self.output_dataset
        ds.constant_wave = self.input_constant_wave

        if self.input_constant_wave:
            self.output_dataset.set_wave(self.input_datasets[0].wave,
                                         self.input_datasets[0].wave_edges)

        self.output_dataset.allocate_values(spectrum_count=self.input_data_count, wave_count=self.input_wave_count)

    def process_item(self, i):
        # Take the spectrum from the first dataset and
        # merge the rest incrementally

        spec = None
        for ds in self.input_datasets:
            if spec is None:
                spec = ds.get_spectrum(i)
            else:
                s = ds.get_spectrum(i)

                # TODO: consider moving these under the Spectrum class
                if self.operation == 'sum':
                    spec.flux += s.flux

                    if spec.flux_err is not None and s.flux_err is not None:
                        spec.flux_err = np.sqrt(spec.flux_err**2 + s.flux_err**2)

                    if spec.flux_sky is not None and s.flux_sky is not None:
                        spec.flux_sky += s.flux_sky

                    if spec.cont is not None and s.cont is not None:
                        spec.cont += s.cont

                    if spec.cont_fit is not None and s.cont_fit is not None:
                        spec.cont_fit += s.cont_fit

                    if spec.mask is not None and s.mask is not None:
                        spec.merge_mask(s.mask)

        if not self.output_dataset.constant_wave:
            self.output_dataset.set_wave(ds.wave, i)

        self.output_dataset.set_flux(spec.flux, i)

        if spec.flux_err is not None:
            self.output_dataset.set_error(spec.flux_err, i)
        
        if spec.mask is not None:
            self.output_dataset.set_mask(spec.mask, i)

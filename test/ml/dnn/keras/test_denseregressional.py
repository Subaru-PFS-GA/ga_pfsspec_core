import os
import numpy as np
from test.test_base import TestBase

from pfsspec.data.dataset import Dataset
from pfsspec.ml.dnn.keras.denseregressional import DenseRegressional
from pfsspec.surveys.sdssaugmenter import SdssAugmenter

class TestDenseRegressional(TestBase):
    def test_train(self):
        dataset = Dataset()
        dataset.load(os.path.join(self.PFSSPEC_DATA_PATH, 'pfs_spec_test', 'sdss_test', 'dataset.dat'))
        dataset.flux = dataset.flux[:1000]
        dataset.params = dataset.params[:1000]
        split_index, ts, vs = dataset.split(0.9)

        labels = ['log_g', ]
        coeffs = [1, ]

        training_augmenter = SdssAugmenter(ts, labels, coeffs, batch_size=200)
        validation_augmenter = SdssAugmenter(vs, labels, coeffs, batch_size=200)

        model = DenseRegressional(levels=1, units=32)
        model.epochs = 1
        model.set_model_shapes(training_augmenter.input_shape, training_augmenter.output_shape)
        model.ensure_model_created()
        model.train(training_augmenter, validation_augmenter)
        model.load_weights(model.checkpoint_path)      # TODO
        model.predict(validation_augmenter)


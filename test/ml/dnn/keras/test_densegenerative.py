import os
import numpy as np
from test.test_base import TestBase

from pfsspec.data.dataset import Dataset
from pfsspec.ml.dnn.keras.densegenerative import DenseGenerative
from pfsspec.stellarmod.kuruczgenerativeaugmenter import KuruczGenerativeAugmenter
from pfsspec.ml.dnn.keras.losses import *

class TestDenseGenerative(TestBase):
    def test_train(self):
        dataset = Dataset()
        dataset.load(os.path.join(self.PFSSPEC_DATA_PATH, 'pfs_spec_test', 'kurucz_test', 'dataset.dat'))
        dataset.flux = dataset.flux[:1000]
        dataset.params = dataset.params[:1000]
        split_index, ts, vs = dataset.split(0.9)

        labels = ['t_eff', 'fe_h', 'log_g', ]
        coeffs = [1000, 1, 1]

        training_generator = KuruczGenerativeAugmenter(ts, labels, coeffs, batch_size=200)
        validation_generator = KuruczGenerativeAugmenter(vs, labels, coeffs, batch_size=200)

        model = DenseGenerative(levels=1, units=32)
        model.loss = 'max_absolute_error'
        model.epochs = 1
        model.set_model_shapes(training_generator.input_shape, training_generator.output_shape)
        model.ensure_model_created()
        model.train(training_generator, validation_generator)
        model.load_weights(model.checkpoint_path)   # TODO
        model.predict(validation_generator)

    def test_train_nosplit(self):
        dataset = Dataset()
        dataset.load(os.path.join(self.PFSSPEC_DATA_PATH, 'pfs_spec_test', 'kurucz_test', 'dataset.dat'))
        dataset.flux = dataset.flux[:1000]
        dataset.params = dataset.params[:1000]
        split_index, ts, vs = dataset.split(0.9)

        labels = ['t_eff', 'fe_h', 'log_g', ]
        coeffs = [1000, 1, 1]

        generator = KuruczGenerativeAugmenter(ts, labels, coeffs, batch_size=200, shuffle=True)
        training_generator = type(generator)(generator)
        validation_generator = type(generator)(generator)

        model = DenseGenerative(levels=1, units=32)
        model.loss = 'max_absolute_error'
        model.epochs = 1
        model.ensure_model_created(training_generator.input_shape, training_generator.output_shape)
        model.train(training_generator, validation_generator)
        model.load_weights(model.checkpoint_path)      # TODO
        model.predict(validation_generator)
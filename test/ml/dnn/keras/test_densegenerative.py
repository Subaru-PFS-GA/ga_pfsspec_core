import os
import numpy as np
from test.test_base import TestBase

from pfsspec.data.dataset import Dataset
from pfsspec.ml.dnn.keras.densegenerative import DenseGenerative
from pfsspec.stellarmod.modelspectrumgenerativeaugmenter import ModelSpectrumGenerativeAugmenter
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

        training_augmenter = ModelSpectrumGenerativeAugmenter(ts, labels, coeffs, batch_size=200)
        validation_augmenter = ModelSpectrumGenerativeAugmenter(vs, labels, coeffs, batch_size=200)

        model = DenseGenerative(levels=1, units=32)
        model.loss = 'max_absolute_error'
        model.epochs = 1
        model.set_model_shapes(training_augmenter.input_shape, training_augmenter.output_shape)
        model.ensure_model_created()
        model.train(training_augmenter, validation_augmenter)
        model.load_weights(model.checkpoint_path)   # TODO
        model.predict(validation_augmenter)

    def test_train_nosplit(self):
        dataset = Dataset()
        dataset.load(os.path.join(self.PFSSPEC_DATA_PATH, 'pfs_spec_test', 'kurucz_test', 'dataset.dat'))
        dataset.flux = dataset.flux[:1000]
        dataset.params = dataset.params[:1000]
        split_index, ts, vs = dataset.split(0.9)

        labels = ['t_eff', 'fe_h', 'log_g', ]
        coeffs = [1000, 1, 1]

        augmenter = ModelSpectrumGenerativeAugmenter(ts, labels, coeffs, batch_size=200, shuffle=True)
        training_augmenter = type(augmenter)(augmenter)
        validation_augmenter = type(augmenter)(augmenter)

        model = DenseGenerative(levels=1, units=32)
        model.loss = 'max_absolute_error'
        model.epochs = 1
        model.ensure_model_created(training_augmenter.input_shape, training_augmenter.output_shape)
        model.train(training_augmenter, validation_augmenter)
        model.load_weights(model.checkpoint_path)      # TODO
        model.predict(validation_augmenter)
import os
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
from test.test_base import TestBase

from pfsspec.ml.dnn.keras.losses import *
from pfsspec.ml.dnn.keras.parallelsum import ParallelSum

class TestParallelSum(TestBase):
    def create_model(self):
        inputs = x = kl.Input((100, 1))                 # output shape: [*, 100]
        x = kl.Conv1D(32, 11, padding='same')(x)        # output shape: [*, 100, 32]
        x = ParallelSum()(x)                          # output shape: [*, 32]
        outputs = x = kl.Dense(1)(x)
        return km.Model(inputs=inputs, outputs=x)

    def test_compile(self):
        model = self.create_model()
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    def test_train(self):
        x = np.random.normal(size=(128, 100, 1))
        y = np.random.normal(size=(128))
        model = self.create_model()
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        model.fit(x, y, epochs=1, batch_size=128)
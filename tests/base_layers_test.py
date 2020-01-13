import tensorflow as tf

from dynastes.layers import DynastesDense
from dynastes.util.test_utils import layer_test


class DynastesDenseTest1D(tf.test.TestCase):
    def test_simple(self):
        layer_test(
            DynastesDense, kwargs={'units': 12}, input_shape=(5, 32, 3))

    def test_specnorm(self):
        layer_test(
            DynastesDense, kwargs={'units': 12,
                                   'kernel_normalizer': 'spectral',
                                   'use_wscale': True,
                                   'kernel_regularizer': 'orthogonal'}, input_shape=(5, 32, 3))

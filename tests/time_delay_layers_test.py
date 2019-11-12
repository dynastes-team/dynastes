import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.keras.testing_utils import layer_test

from dynastes.layers.time_delay_layers import TimeDelayLayer1D, DepthGroupwiseTimeDelayLayer1D, \
    DepthGroupwiseTimeDelayLayerFake2D, TimeDelayLayerFake2D


class TimeDelayLayer1DTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope({'TimeDelayLayer1D': TimeDelayLayer1D}):
            layer_test(
                TimeDelayLayer1D, kwargs={'filters': 4}, input_shape=(5, 32, 3))

    def test_specnorm(self):
        with custom_object_scope({'TimeDelayLayer1D': TimeDelayLayer1D}):
            layer_test(
                TimeDelayLayer1D, kwargs={'filters': 4,
                                          'kernel_normalizer': 'spectral',
                                          'kernel_regularizer': 'orthogonal'}, input_shape=(5, 32, 3))


class SeparableTimeDelayLayer1DTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope(
                {'DepthGroupwiseTimeDelayLayer1D': DepthGroupwiseTimeDelayLayer1D,
                 'TimeDelayLayer1D': TimeDelayLayer1D}):
            layer_test(
                DepthGroupwiseTimeDelayLayer1D, kwargs={'depth_multiplier': 2}, input_shape=(5, 32, 3))


class SeparableTimeDelayLayerFake2DTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope({'DepthGroupwiseTimeDelayLayerFake2D': DepthGroupwiseTimeDelayLayerFake2D}):
            layer_test(
                DepthGroupwiseTimeDelayLayerFake2D, input_shape=(5, 16, 16, 3))


class TimeDelayLayerFake2DTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope({'TimeDelayLayerFake2D': TimeDelayLayerFake2D}):
            layer_test(
                TimeDelayLayerFake2D, kwargs={'filters': 4}, input_shape=(5, 16, 16, 3))


if __name__ == '__main__':
    tf.test.main()

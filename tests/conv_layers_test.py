import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.keras.testing_utils import layer_test

from dynastes import object_scope
from dynastes.layers.convolutional_layers import DynastesConv1DTranspose, DynastesDepthwiseConv1D, \
    DynastesConv2DTranspose, DynastesConv2D, DynastesDepthwiseConv2D, DynastesConv1D


class DynastesConv1DTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesConv1D, kwargs={'filters': 3, 'kernel_size': 3}, input_shape=(5, 32, 3))

    def test_specnorm(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesConv1D, kwargs={'filters': 3, 'kernel_size': 3,
                                        'kernel_normalizer': 'spectral',
                                        'kernel_regularizer': 'orthogonal'}, input_shape=(5, 32, 3))


class DynastesConv2DTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesConv2D, kwargs={'filters': 3, 'kernel_size': (3, 3)}, input_shape=(4, 16, 16, 3))

    def test_specnorm(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesConv2D, kwargs={'filters': 7, 'kernel_size': (3, 3),
                                        'kernel_normalizer': 'spectral',
                                        'kernel_regularizer': 'orthogonal'}, input_shape=(4, 16, 16, 5))


class DynastesConv1DTransposeTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesConv1DTranspose,
                kwargs={'filters': 3, 'kernel_size': 3, 'strides': 2, 'padding': 'same'},
                input_shape=(None, 16, 3),
                expected_output_shape=(None, 32, 3)
            )

    def test_specnorm(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesConv1DTranspose,
                kwargs={'filters': 7, 'kernel_size': 3, 'strides': 2, 'padding': 'same',
                        'kernel_normalizer': 'spectral',
                        'kernel_regularizer': 'orthogonal'},
                input_shape=(None, 16, 5),
                expected_output_shape=(None, 32, 7)
            )


class DynastesConv2DTransposeTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesConv2DTranspose,
                kwargs={'filters': 3, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'same'},
                input_shape=(None, 16, 16, 3),
                expected_output_shape=(None, 32, 32, 3)
            )

    def test_specnorm(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesConv2DTranspose,
                kwargs={'filters': 7, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'same',
                        'kernel_normalizer': 'spectral',
                        'kernel_regularizer': 'orthogonal'},
                input_shape=(None, 16, 16, 5),
                expected_output_shape=(None, 32, 32, 7)
            )


class DynastesDepthwiseConv1DTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesDepthwiseConv1D, kwargs={'kernel_size': 3}, input_shape=(5, 32, 3))

    def test_specnorm(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesDepthwiseConv1D, kwargs={'kernel_size': 3,
                                                 'kernel_normalizer': 'spectral',
                                                 'kernel_regularizer': 'orthogonal'}, input_shape=(5, 32, 3))


class DynastesDepthwiseConv2DTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesDepthwiseConv2D, kwargs={'kernel_size': (3, 3)}, input_shape=(4, 16, 16, 3))

    def test_specnorm(self):
        with custom_object_scope(object_scope):
            layer_test(
                DynastesDepthwiseConv2D, kwargs={'kernel_size': (3, 3),
                                                 'kernel_normalizer': 'spectral'}, input_shape=(4, 16, 16, 5))

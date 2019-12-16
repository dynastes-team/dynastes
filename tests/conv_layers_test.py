import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope

from dynastes import object_scope
from dynastes.layers.convolutional_layers import DynastesConv1DTranspose, DynastesDepthwiseConv1D, \
    DynastesConv2DTranspose, DynastesConv2D, DynastesDepthwiseConv2D, DynastesConv1D
from dynastes.util.test_utils import layer_test

import numpy as np
to_tensor = tf.convert_to_tensor
normal = np.random.normal

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
                                        'use_wscale': True,
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

    def test_masking(self):
        with custom_object_scope(object_scope):

            layer = DynastesConv1DTranspose(32, kernel_size=3, strides=2, padding='same')

            ts = to_tensor(normal(size=(8, 16, 32))
                          .astype(np.float32))

            mask_len = 16 // 6
            mask = to_tensor(([True] * (16 - mask_len)) + ([False] * (mask_len)))
            mask = tf.expand_dims(mask, axis=0)
            mask = tf.tile(mask, [8, 1])
            print(mask.shape)
            layer(ts, mask=mask)
            layer.compute_mask(ts, mask=mask)


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

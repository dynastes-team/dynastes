import numpy as np
import tensorflow as tf
import dynastes.layers as dynl
from dynastes.layers.convolutional_layers import DynastesConv1DTranspose, DynastesDepthwiseConv1D, \
    DynastesConv2DTranspose, DynastesConv2D, DynastesDepthwiseConv2D, DynastesConv1D, Upsampling1D, Upsampling2D
from dynastes.util.test_utils import layer_test

to_tensor = tf.convert_to_tensor
normal = np.random.normal


class DynastesConv1DTest(tf.test.TestCase):

    def test_masking(self):
        layer = DynastesConv1D(32, kernel_size=3, strides=2, padding='same')

        ts = to_tensor(normal(size=(8, 16, 32))
                       .astype(np.float32))

        mask_len = 16 // 6
        mask = to_tensor(([True] * (16 - mask_len)) + ([False] * (mask_len)))
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [8, 1])
        layer(ts, mask=mask)
        layer.compute_mask(ts, mask=mask)

        @tf.function
        def graph_test_fn(x, mask):
            layer(x, mask=mask)
            layer.compute_mask(x, mask=mask)

        graph_test_fn(x=ts, mask=mask)

    def test_simple(self):
        layer_test(
            DynastesConv1D, kwargs={'filters': 3, 'kernel_size': 3, 'activation': 'mish'}, input_shape=(5, 32, 3))

    def test_specnorm(self):
        layer_test(
            DynastesConv1D, kwargs={'filters': 3, 'kernel_size': 3,
                                    'kernel_normalizer': 'spectral',
                                    'use_wscale': True,
                                    'kernel_regularizer': 'orthogonal'}, input_shape=(5, 32, 3))


class DynastesConv2DTest(tf.test.TestCase):
    def test_simple(self):
        layer_test(
            DynastesConv2D, kwargs={'filters': 3, 'kernel_size': (3, 3), 'activation': 'swish'},
            input_shape=(4, 16, 16, 3))

    def test_masking(self):
        layer = DynastesConv2D(32, kernel_size=3, strides=2, padding='same')

        ts = to_tensor(normal(size=(8, 16, 16, 32))
                       .astype(np.float32))

        mask_len = 16 // 8
        mask = to_tensor(([True] * (16 - mask_len)) + ([False] * (mask_len)))
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [8, 1])
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, [1, 16, 1])
        layer(ts, mask=mask)
        layer.compute_mask(ts, mask=mask)

        @tf.function
        def graph_test_fn(x, mask):
            layer(x, mask=mask)
            layer.compute_mask(x, mask=mask)

        graph_test_fn(x=ts, mask=mask)

    def test_specnorm(self):
        layer_test(
            DynastesConv2D, kwargs={'filters': 7, 'kernel_size': (3, 3),
                                    'kernel_normalizer': 'spectral',
                                    'kernel_regularizer': 'orthogonal'}, input_shape=(4, 16, 16, 5))


class DynastesConv1DTransposeTest(tf.test.TestCase):
    def test_simple(self):
        layer_test(
            DynastesConv1DTranspose,
            kwargs={'filters': 3, 'kernel_size': 3, 'strides': 2, 'padding': 'same'},
            input_shape=(None, 16, 3),
            expected_output_shape=(None, 32, 3)
        )

    def test_masking(self):
        layer = DynastesConv1DTranspose(32, kernel_size=3, strides=2, padding='same')

        ts = to_tensor(normal(size=(8, 16, 32))
                       .astype(np.float32))

        mask_len = 16 // 6
        mask = to_tensor(([True] * (16 - mask_len)) + ([False] * (mask_len)))
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [8, 1])
        layer(ts, mask=mask)
        layer.compute_mask(ts, mask=mask)

        @tf.function
        def graph_test_fn(x, mask):
            layer(x, mask=mask)
            layer.compute_mask(x, mask=mask)

        graph_test_fn(x=ts, mask=mask)

    def test_specnorm(self):
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
        layer_test(
            DynastesConv2DTranspose,
            kwargs={'filters': 3, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'same'},
            input_shape=(None, 16, 16, 3),
            expected_output_shape=(None, 32, 32, 3)
        )

    def test_masking(self):
        layer = DynastesConv2DTranspose(32, kernel_size=3, strides=2, padding='same')

        ts = to_tensor(normal(size=(8, 16, 16, 32))
                       .astype(np.float32))

        mask_len = 16 // 8
        mask = to_tensor(([True] * (16 - mask_len)) + ([False] * (mask_len)))
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [8, 1])
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, [1, 16, 1])
        layer(ts, mask=mask)
        layer.compute_mask(ts, mask=mask)

        @tf.function
        def graph_test_fn(x, mask):
            layer(x, mask=mask)
            layer.compute_mask(x, mask=mask)

        graph_test_fn(x=ts, mask=mask)

    def test_specnorm(self):
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
        layer_test(
            DynastesDepthwiseConv1D, kwargs={'kernel_size': 3}, input_shape=(5, 32, 3))

    def test_masking(self):
        layer = DynastesDepthwiseConv1D(kernel_size=3, strides=2, padding='same')

        ts = to_tensor(normal(size=(8, 16, 32))
                       .astype(np.float32))

        mask_len = 16 // 6
        mask = to_tensor(([True] * (16 - mask_len)) + ([False] * (mask_len)))
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [8, 1])
        layer(ts, mask=mask)
        layer.compute_mask(ts, mask=mask)

        @tf.function
        def graph_test_fn(x, mask):
            layer(x, mask=mask)
            layer.compute_mask(x, mask=mask)

        graph_test_fn(x=ts, mask=mask)

    def test_specnorm(self):
        layer_test(
            DynastesDepthwiseConv1D, kwargs={'kernel_size': 3,
                                             'kernel_normalizer': 'spectral',
                                             'kernel_regularizer': 'orthogonal'}, input_shape=(5, 32, 3))


class Upsampling2DTest(tf.test.TestCase):
    def test_simple(self):
        layer_test(
            Upsampling2D,
            kwargs={'strides': (2, 2)},
            input_shape=(None, 16, 16, 3),
            expected_output_shape=(None, 32, 32, 3)
        )

    def test_masking(self):
        layer = Upsampling2D(strides=2)

        ts = to_tensor(normal(size=(8, 16, 16, 32))
                       .astype(np.float32))

        mask_len = 16 // 8
        mask = to_tensor(([True] * (16 - mask_len)) + ([False] * (mask_len)))
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [8, 1])
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, [1, 16, 1])
        layer(ts, mask=mask)
        layer.compute_mask(ts, mask=mask)

        @tf.function
        def graph_test_fn(x, mask):
            layer(x, mask=mask)
            layer.compute_mask(x, mask=mask)

        graph_test_fn(x=ts, mask=mask)


class Upsampling1DTest(tf.test.TestCase):
    def test_simple(self):
        layer_test(
            Upsampling1D, kwargs={'strides': 2}, input_shape=(None, 32, 3), expected_output_shape=(None, 64, 3))

    def test_masking(self):
        layer = Upsampling1D(strides=2)

        ts = to_tensor(normal(size=(8, 16, 32))
                       .astype(np.float32))

        mask_len = 16 // 6
        mask = to_tensor(([True] * (16 - mask_len)) + ([False] * (mask_len)))
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [8, 1])
        layer(ts, mask=mask)
        layer.compute_mask(ts, mask=mask)

        @tf.function
        def graph_test_fn(x, mask):
            layer(x, mask=mask)
            layer.compute_mask(x, mask=mask)

        graph_test_fn(x=ts, mask=mask)


class DynastesDepthwiseConv2DTest(tf.test.TestCase):
    def test_simple(self):
        layer_test(
            DynastesDepthwiseConv2D, kwargs={'kernel_size': (3, 3)}, input_shape=(4, 16, 16, 3))

    def test_masking(self):
        layer = DynastesDepthwiseConv2D(kernel_size=3, strides=2, padding='same')

        ts = to_tensor(normal(size=(8, 16, 16, 32))
                       .astype(np.float32))

        mask_len = 16 // 6
        mask = to_tensor(([True] * (16 - mask_len)) + ([False] * (mask_len)))
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [8, 1])
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, [1, 16, 1])
        layer(ts, mask=mask)
        layer.compute_mask(ts, mask=mask)

        @tf.function
        def graph_test_fn(x, mask):
            layer(x, mask=mask)
            layer.compute_mask(x, mask=mask)

        graph_test_fn(x=ts, mask=mask)

    def test_specnorm(self):
        layer_test(
            DynastesDepthwiseConv2D, kwargs={'kernel_size': (3, 3),
                                             'kernel_normalizer': 'spectral'}, input_shape=(4, 16, 16, 5))

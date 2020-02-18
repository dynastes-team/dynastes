import tensorflow as tf

from dynastes.layers import DynastesDense, DynastesEmbedding
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
                                   'wnorm': True,
                                   'kernel_regularizer': 'orthogonal'}, input_shape=(5, 32, 3))


class DynastesEmbeddingTest(tf.test.TestCase):

    def test_grads(self):
        emb = DynastesEmbedding(4, 8, mask_zero=True, input_length=1)
        inps = tf.convert_to_tensor([[1]], dtype=tf.int32)
        targ = tf.random.normal([1, 1, 8], dtype=tf.float32)

        @tf.function
        def grads_fn(inps, targ):
            with tf.GradientTape() as t:
                t.watch(emb.trainable_weights)
                r = emb(inps, training=True)
                l = tf.reduce_sum(tf.keras.losses.mean_squared_error(targ, r))
            grads = t.gradient(l, emb.trainable_weights)
            return grads

        grads = grads_fn(inps, targ)
        c_grad = tf.clip_by_global_norm(grads, clip_norm=15.)
        print(grads[0], c_grad)
        assert grads[0] is not None

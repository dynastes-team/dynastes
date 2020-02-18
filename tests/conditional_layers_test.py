import tensorflow as tf

from dynastes.layers import EmbeddingKernelDense
from dynastes.util.test_utils import layer_test


class DynastesEmbeddingTest(tf.test.TestCase):

    def test_grads(self):
        emb = EmbeddingKernelDense(16, 5)
        input_dense = tf.random.normal([2, 7, 3, 8], dtype=tf.float32)
        input_indices = tf.convert_to_tensor([[1],[2]], dtype=tf.int64)
        inps = [input_dense, input_indices]
        targ = tf.random.normal([2, 7, 3, 16], dtype=tf.float32)

        r = emb(inps, training=True)
        self.assertShapeEqual(targ.numpy(), r)
        print(r.shape)
        @tf.function
        def grads_fn(inps, targ):
            with tf.GradientTape() as t:
                t.watch(emb.trainable_weights)
                r = emb(inps, training=True)
                l = tf.reduce_sum(tf.keras.losses.mean_squared_error(targ, r))
            grads = t.gradient(l, emb.trainable_weights)
            return grads
        grads = grads_fn(inps, targ)
        assert grads[0] is not None
import tensorflow as tf

from dynastes.layers import EmbeddingKernelDense
from dynastes.util.test_utils import layer_test


class DynastesEmbeddingTest(tf.test.TestCase):

    def test_grads(self):
        emb = EmbeddingKernelDense(16, 5)
        input_dense = tf.random.normal([2, 2, 7, 8], dtype=tf.float32)
        input_indices = tf.convert_to_tensor([[1,2],[3,4]], dtype=tf.int32)
        inps = [input_dense, input_indices]
        targ = tf.random.normal([2, 2, 7, 16], dtype=tf.float32)
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
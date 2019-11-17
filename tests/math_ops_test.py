import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.framework import test_util

import dynastes as d
from dynastes.core import math_ops
from dynastes.core.nn.math_ops import _lsh_similarity_grad_forward, _lsh_similarity

class LshSimilarityTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):

        n_items = 16

        depth = 8
        bucket_length = 32

        x = tf.convert_to_tensor(np.random.normal(size=(n_items, depth)).astype(np.float32))
        y = tf.convert_to_tensor(np.random.normal(size=(n_items, depth)).astype(np.float32))

        def _idx_to_bits(i, bucket_length):
            """Convert an group index to its bit representation."""
            bits = bin(i)[2:].zfill(bucket_length)  # Pad the bits str with 0
            return [-1.0 if b == "0" else 1.0 for b in bits]

        with tf.compat.v1.variable_scope("lsh_gating"):
            # Vectors defining the hyperplanes
            t_vectors = tf.compat.v1.get_variable(
                "vector",
                shape=(depth, bucket_length*8),
                dtype=tf.float16,
                trainable=False,
            )
            # Projection vector from the bit space to similarity score space
            t_group = tf.compat.v1.constant(
                [_idx_to_bits(i, bucket_length*8) for i in range(bucket_length)],
                dtype=tf.float16,
                name="group")

        real_fn_out = math_ops.lsh_similarity(x, y, t_vectors=t_vectors, t_group=t_group, bucket_length=bucket_length, threshold=0.00000001)
        grad_fn_out = math_ops.lsh_similary_back_fn(x, y, t_vectors=t_vectors, t_group=t_group, bucket_length=bucket_length)

        self.assertAlmostEqual(0., tf.reduce_mean(real_fn_out - grad_fn_out).numpy(), places=2)

        with tf.GradientTape() as tape:
            tape.watch(x)
            tape.watch(y)
            r = math_ops.lsh_similarity(x, y, t_vectors=t_vectors, t_group=t_group, bucket_length=bucket_length)
            loss = tf.keras.losses.mse(tf.eye(n_items), r)

        dx, dy = tape.gradient(loss, [x, y])

    @test_util.use_deterministic_cudnn
    def test_attn(self):
        n_items = 16

        batch = 3
        heads = 4


        depth = 64
        nb_hyperplanes = 64
        nb_buckets = 64

        q = tf.convert_to_tensor(np.random.normal(size=(batch, heads, n_items, depth)).astype(np.float32))
        k = tf.convert_to_tensor(np.random.normal(size=(batch, 1, n_items, depth)).astype(np.float32))
        v = tf.convert_to_tensor(np.random.normal(size=(batch, 1, n_items, depth)).astype(np.float32))
        t = tf.convert_to_tensor(np.random.normal(size=(batch, heads, n_items, depth)).astype(np.float32))

        def _idx_to_bits(i, nb_hyperplanes):
            """Convert an group index to its bit representation."""
            bits = bin(i)[2:].zfill(nb_hyperplanes)  # Pad the bits str with 0
            return [-1.0 if b == "0" else 1.0 for b in bits]

        with tf.compat.v1.variable_scope("lsh_gating"):
            # Vectors defining the hyperplanes
            t_vectors = tf.compat.v1.get_variable(
                "vector",
                shape=(depth, nb_hyperplanes),
                dtype=tf.float16,
                trainable=False,
            )
            # Projection vector from the bit space to similarity score space
            t_group = tf.compat.v1.constant(
                [_idx_to_bits(i, nb_hyperplanes) for i in range(nb_buckets)],
                dtype=tf.float16,
                name="group")

        real_fn_out = math_ops.lsh_attention(q, k, v, t_vectors_q=t_vectors, t_group=t_group, bucket_length=nb_hyperplanes, threshold=0.000000001)
        grad_fn_out = math_ops.lsh_attention_back_fn(q, k, v, t_vectors_q=t_vectors, t_group=t_group,
                                                     bucket_length=nb_hyperplanes)
        self.assertAlmostEqual(0., tf.reduce_mean(real_fn_out - grad_fn_out).numpy(), places=2)

        @tf.function
        def test_func(_q, _k, _v):
            return math_ops.lsh_attention(_q, _k, _v, t_vectors_q=t_vectors, t_group=t_group, bucket_length=nb_hyperplanes, threshold=0.000000001)

        #r_func = test_func(q,k,v)

        with tf.GradientTape() as tape:
            tape.watch(q)
            tape.watch(k)
            tape.watch(v)
            r = math_ops.lsh_attention(q, k, v, t_vectors_q=t_vectors, t_group=t_group, bucket_length=nb_hyperplanes)
            loss = tf.keras.losses.mse(t, r)

        dq, dk, dv = tape.gradient(loss, [q, k, v])




if __name__ == '__main__':
    tf.test.main()
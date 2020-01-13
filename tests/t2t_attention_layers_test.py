import timeit

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.framework import test_util

import dynastes as d
from dynastes.layers.t2t_attention_layers import Attention1D, Attention2D, PseudoBlockSparseAttention1D
from dynastes.probability.pseudoblocksparse_bijectors import BlockSparseStridedRoll1D


def _test_grads(testCase: tf.test.TestCase, func, input):
    _, grads = tf.test.compute_gradient(func, input)
    for grad in grads:
        testCase.assertNotAllClose(grad, np.zeros_like(grad))
        testCase.assertAllInRange(grad, -400., 400)


to_tensor = tf.convert_to_tensor
normal = np.random.normal


class T2TAttention1DTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        tf.config.optimizer.set_jit(True)
        t_steps = 12
        bs = 2
        dim = 16
        dim_mq = 32
        num_heads = 4
        s = 2

        layers = [

            (
                'PsuedoBlockSparse Masked',
                PseudoBlockSparseAttention1D(num_heads=num_heads, block_size=8,
                                             blocksparse_bijector=BlockSparseStridedRoll1D(block_size=8),
                                             mask_right=True),
                {'self': True, 'steps_q': 64, 'steps_kv': 64, 'dim_q': dim, 'dim_k': dim, 'dim_v': dim}),
            (
                'PsuedoBlockSparse Multiquery Masked',
                PseudoBlockSparseAttention1D(num_heads=num_heads, block_size=8,
                                             multiquery_attention=True,
                                             blocksparse_bijector=BlockSparseStridedRoll1D(block_size=8),
                                             mask_right=True),
                {'self': True, 'steps_q': 64, 'steps_kv': 64, 'dim_q': dim_mq, 'dim_k': dim_mq // num_heads,
                 'dim_v': dim_mq // num_heads}),
            (
                'Local Masked',
                Attention1D(num_heads=num_heads, self_attention=True, local=True, masked=True, block_length=8,
                            filter_width=6),
                {'self': True, 'steps_q': 64, 'steps_kv': 64, 'dim_q': dim, 'dim_k': dim, 'dim_v': dim}),
            (
                'Local Multiquery Masked',
                Attention1D(num_heads=num_heads, multiquery_attention=True, self_attention=True, local=True,
                            masked=True, block_length=8,
                            filter_width=6),
                {'self': True, 'steps_q': 64, 'steps_kv': 64, 'dim_q': dim_mq, 'dim_k': dim_mq // num_heads,
                 'dim_v': dim_mq // num_heads}),
            (
                'Sparse Unmasked',
                Attention1D(num_heads=num_heads, self_attention=True, sparse=True, lsh_bucket_length=3,
                            mask_right=True),
                {'self': True, 'steps_q': 32, 'steps_kv': 32, 'dim_q': dim, 'dim_k': dim, 'dim_v': dim}),
            (
                'Normal',
                Attention1D(num_heads=num_heads, self_attention=False),
                {'self': False, 'steps_q': 16, 'steps_kv': 64, 'dim_q': dim, 'dim_k': dim, 'dim_v': dim}
            ),
            (
                'Normal Multiquery',
                Attention1D(num_heads=num_heads, multiquery_attention=True, self_attention=False),
                {'self': False, 'steps_q': 16, 'steps_kv': 64, 'dim_q': dim_mq, 'dim_k': dim_mq // num_heads,
                 'dim_v': dim_mq // num_heads}
            ),
            (
                'Relative',
                Attention1D(num_heads=num_heads, self_attention=True, relative=True, max_relative_position=8,
                            mask_right=True),
                {'self': True, 'steps_q': 16, 'steps_kv': 16, 'dim_q': dim, 'dim_k': dim, 'dim_v': dim}
            ),
            (
                'Relative Multiquery',
                Attention1D(num_heads=num_heads, multiquery_attention=True, self_attention=True, relative=True,
                            max_relative_position=8,
                            mask_right=True),
                {'self': True, 'steps_q': 32, 'steps_kv': 32, 'dim_q': dim_mq, 'dim_k': dim_mq // num_heads,
                 'dim_v': dim_mq // num_heads}
            ),
            (
                'Relative Masked',
                Attention1D(num_heads=num_heads, self_attention=True, relative=True, max_relative_position=8,
                            masked=True,
                            mask_right=True),
                {'self': True, 'steps_q': 16, 'steps_kv': 16, 'dim_q': dim, 'dim_k': dim, 'dim_v': dim}
            ),
            (
                'Relative Multiquery Masked',
                Attention1D(num_heads=num_heads, multiquery_attention=True,
                            self_attention=True, relative=True, max_relative_position=8,
                            masked=True,
                            mask_right=True),
                {'self': True, 'steps_q': 16, 'steps_kv': 16, 'dim_q': dim_mq, 'dim_k': dim_mq // num_heads,
                 'dim_v': dim_mq // num_heads}
            ),
            (
                'Local',
                Attention1D(num_heads=num_heads, self_attention=True, local=True, block_length=8, filter_width=6),
                {'self': True, 'steps_q': 64, 'steps_kv': 64, 'dim_q': dim, 'dim_k': dim, 'dim_v': dim}),
            (
                'Local Multiquery',
                Attention1D(num_heads=num_heads, multiquery_attention=True, self_attention=True, local=True,
                            block_length=8, filter_width=6),
                {'self': True, 'steps_q': 64, 'steps_kv': 64, 'dim_q': dim_mq, 'dim_k': dim_mq // num_heads,
                 'dim_v': dim_mq // num_heads}),

        ]

        def test_layer(layer, params):
            q = to_tensor(normal(size=(bs, params['steps_q'], params['dim_q']))
                          .astype(np.float32))
            k = to_tensor(normal(size=(bs, params['steps_kv'], params['dim_k']))
                          .astype(np.float32))
            v = to_tensor(normal(size=(bs, params['steps_kv'], params['dim_v']))
                          .astype(np.float32))
            mask_len = params['steps_q'] // 6
            mask_q = to_tensor(([True] * (params['steps_q'] - mask_len)) + ([False] * (mask_len)))
            mask_q = tf.expand_dims(mask_q, axis=0)
            mask_q = tf.tile(mask_q, [bs, 1])

            mask_kv = to_tensor(([True] * (params['steps_kv'] - mask_len)) + ([False] * (mask_len)))
            mask_kv = tf.expand_dims(mask_kv, axis=0)
            mask_kv = tf.tile(mask_kv, [bs, 1])

            def get_test_fn(layer, mask):
                @tf.function
                def test_func(q, k, v):
                    r, _ = layer([q, k, v], mask=mask)
                    return r

                return test_func

            mask = [mask_q, mask_kv, mask_kv]
            r, _ = layer([q, k, v], mask=mask)

            test_fn = get_test_fn(layer, mask)
            # Warmup
            rfn = test_fn(q, k=k, v=v)
            self.assertAllClose(r, rfn)
            self.assertAllInRange(r, -200., 200)

            def fn():
                test_fn(q, k=k, v=v)

            time = timeit.timeit(fn, number=2) / (params['steps_q'] * params['steps_kv'] * params['dim_q'])
            time *= 8192

            _test_grads(self, test_fn, [q, k, v])
            return time

        for (type, layer, params) in layers:
            print(type, test_layer(layer, params))


class T2TAttention2DTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        tf.config.optimizer.set_jit(True)
        t_steps = 12
        bs = 2
        dim = 8
        dim_mq = 16
        num_heads = 4
        s = 2

        layers = [

            (
                'Relative',
                Attention2D(num_heads=num_heads, self_attention=True, relative=True, max_relative_position=8,
                            mask_right=True),
                {'self': True, 'steps_q': 8, 'steps_kv': 8, 'dim_q': dim, 'dim_k': dim, 'dim_v': dim}
            ),
            (
                'Relative Multiquery',
                Attention2D(num_heads=num_heads, multiquery_attention=True, self_attention=True, relative=True,
                            max_relative_position=8,
                            mask_right=True),
                {'self': True, 'steps_q': 8, 'steps_kv': 8, 'dim_q': dim_mq, 'dim_k': dim_mq // num_heads,
                 'dim_v': dim_mq // num_heads}
            )
        ]

        # TODO: Add these back but for 2D
        """
        (
            'Local',
            Attention1D(num_heads=num_heads, self_attention=True, local=True, block_length=8, filter_width=6),
            {'self': True, 'steps_q': 64, 'steps_kv': 64, 'dim_q': dim, 'dim_k': dim, 'dim_v': dim}),
        (
            'Local Multiquery',
            Attention1D(num_heads=num_heads, multiquery_attention=True, self_attention=True, local=True,
                        block_length=8, filter_width=6),
            {'self': True, 'steps_q': 64, 'steps_kv': 64, 'dim_q': dim_mq, 'dim_k': dim_mq // num_heads,
             'dim_v': dim_mq // num_heads}),
        (
            'Local Masked',
            Attention1D(num_heads=num_heads, self_attention=True, local=True, masked=True, block_length=8,
                        filter_width=6),
            {'self': True, 'steps_q': 64, 'steps_kv': 64, 'dim_q': dim, 'dim_k': dim, 'dim_v': dim}),
        (
            'Local Multiquery Masked',
            Attention1D(num_heads=num_heads, multiquery_attention=True, self_attention=True, local=True,
                        masked=True, block_length=8,
                        filter_width=6),
            {'self': True, 'steps_q': 64, 'steps_kv': 64, 'dim_q': dim_mq, 'dim_k': dim_mq // num_heads,
             'dim_v': dim_mq // num_heads}),
         """

        def test_layer(layer, params):
            q = to_tensor(normal(size=(bs, params['steps_q'], params['steps_q'], params['dim_q']))
                          .astype(np.float32))
            k = to_tensor(normal(size=(bs, params['steps_kv'], params['steps_kv'], params['dim_k']))
                          .astype(np.float32))
            v = to_tensor(normal(size=(bs, params['steps_kv'], params['steps_kv'], params['dim_v']))
                          .astype(np.float32))

            # mask_q = to_tensor(([True] * (params['steps_q'] - 3)) + ([False] * (3)))
            # mask_q = tf.expand_dims(mask_q, axis=0)
            # mask_q = tf.tile(mask_q, [bs, 1])

            # mask_kv = to_tensor(([True] * (params['steps_kv'] - 3)) + ([False] * (3)))
            # mask_kv = tf.expand_dims(mask_kv, axis=0)
            # mask_kv = tf.tile(mask_kv, [bs, 1])

            def get_test_fn(layer, mask):
                @tf.function
                def test_func(q, k, v):
                    r, _ = layer([q, k, v], mask=mask)
                    return r

                return test_func

            # mask = [mask_q, mask_kv, mask_kv]
            mask = None
            r, _ = layer([q, k, v], mask=mask)
            test_fn = get_test_fn(layer, mask)
            # Warmup
            rfn = test_fn(q, k=k, v=v)
            self.assertAllClose(r, rfn)
            self.assertAllInRange(r, -200., 200)

            def fn():
                test_fn(q, k=k, v=v)

            time = timeit.timeit(fn, number=2) / (
                    params['steps_q'] * params['steps_q'] * params['steps_kv'] * params['steps_kv'] * params[
                'dim_q'])
            time *= 8192

            _test_grads(self, test_fn, [q, k, v])
            return time

        for (type, layer, params) in layers:
            print(type, test_layer(layer, params))


if __name__ == '__main__':
    tf.config.optimizer.set_jit(True)
    tf.test.main()

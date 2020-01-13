import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow_addons.layers.normalizations import GroupNormalization

from dynastes.layers.normalization_layers import AdaptiveMultiNormalization, PoolNormalization2D


def _test_grads(testCase: tf.test.TestCase, func, input):
    _, grads = tf.test.compute_gradient(func, input)
    for grad in grads:
        testCase.assertNotAllClose(grad, np.zeros_like(grad))


to_tensor = tf.convert_to_tensor
normal = np.random.normal


class AdaptiveMultiNormalizationTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        normalizers = [
            GroupNormalization(groups=1, center=False, scale=False),
            GroupNormalization(groups=-1, center=False, scale=False),
            PoolNormalization2D(pool_size=(-1, 3))
        ]
        layer = AdaptiveMultiNormalization(layers=normalizers)
        x = tf.convert_to_tensor(normal(size=(1, 8, 8, 8)).astype(np.float16))
        y = tf.convert_to_tensor(normal(size=(1, 2, 3, 4)).astype(np.float16))
        res = layer([x, y])
        self.assertShapeEqual(x.numpy(), res)
        y = tf.convert_to_tensor(normal(size=(1, 4)).astype(np.float16))
        res = layer([x, y])
        self.assertShapeEqual(x.numpy(), res)

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.keras.testing_utils import layer_test

from dynastes.layers.random_layers import StatelessRandomNormalLike


def _test_grads(testCase: tf.test.TestCase, func, input):
    _, grads = tf.test.compute_gradient(func, input)
    for grad in grads:
        testCase.assertNotAllClose(grad, np.zeros_like(grad))


to_tensor = tf.convert_to_tensor
normal = np.random.normal


class StatelessRandomNormalLikeTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        layer_test(
            StatelessRandomNormalLike, input_shape=(5, 32, 3))

        layer = StatelessRandomNormalLike(channels=1)
        input = tf.random.normal(shape=[4, 16, 32, 5])
        seed = tf.convert_to_tensor(np.random.randint(tf.int64.min, tf.int64.max, size=(4, 4), dtype=np.int64))
        res = layer([input, seed])
        self.assertNotAllClose(input, res)
        seed = tf.convert_to_tensor([1, 2, 1, 3])
        res = layer([input, seed])
        self.assertAllEqual(res[0], res[2])

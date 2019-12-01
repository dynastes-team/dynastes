import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.framework import test_util

import dynastes as d
from dynastes.layers.normalization_layers import AdaptiveMultiNormalization, AdaptiveLayerInstanceNormalization
from tensorflow_addons.layers.normalizations import GroupNormalization, InstanceNormalization


def _test_grads(testCase: tf.test.TestCase, func, input):
    _, grads = tf.test.compute_gradient(func, input)
    for grad in grads:
        testCase.assertNotAllClose(grad, np.zeros_like(grad))


to_tensor = tf.convert_to_tensor
normal = np.random.normal


class AdaptiveMultiNormalizationTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            layer = AdaptiveLayerInstanceNormalization()
            x = tf.convert_to_tensor(normal(size=(1, 4, 4, 8)))
            y = tf.convert_to_tensor(normal(size=(1, 4)))

            res = layer([x,y])
            print(res)


import tensorflow as tf
from dynastes.util.test_utils import layer_test
from dynastes.layers.data_augmentation import SpecAugmentLayer

import numpy as np

to_tensor = tf.convert_to_tensor
normal = np.random.normal

class SpecAugmentLayerTest(tf.test.TestCase):
    def test_simple(self):
            layer_test(
                SpecAugmentLayer, kwargs={}, input_shape=(5, 32, 80, 2))

import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope

from dynastes.blocks.attention_blocks import SelfAttentionBlock1D
from dynastes.util.test_utils import layer_test


class SelfAttentionBlock1DTest(tf.test.TestCase):
    def test_simple(self):
            layer_test(
                SelfAttentionBlock1D, kwargs={'attention_dim': 8, 'kernel_size': 3, 'output_dim': 16, 'num_heads': 4,
                                              'multiquery_attention': True}, input_shape=(5, 32, 3))

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.utils import custom_object_scope

from dynastes import object_scope
from dynastes.blocks.attention_blocks import SelfAttentionBlock1D
from dynastes.blocks.transformer_blocks import PointWiseFeedForwardBlock, EncoderBlock, EncoderBlockStack


def _test_grads(testCase: tf.test.TestCase, func, input):
    _, grads = tf.test.compute_gradient(func, input)
    for grad in grads:
        testCase.assertNotAllClose(grad, np.zeros_like(grad))
        testCase.assertAllInRange(grad, -400., 400)


to_tensor = tf.convert_to_tensor
normal = np.random.normal


class EncoderBlockTest(tf.test.TestCase):
    def test_simple(self):
        with custom_object_scope(object_scope):
            d_model = 16
            num_heads = 4
            dff = 32
            length = 4096
            mask_len = 32
            sablock = SelfAttentionBlock1D(d_model, d_model, num_heads=num_heads,
                                           attention_type='PseudoBlockSparseAttention1D',
                                           block_length=1024,
                                           multiquery_attention=True,)
            norm = tfkl.LayerNormalization(epsilon=1e-6)
            df_net = PointWiseFeedForwardBlock(dff=dff, d_model=d_model)
            enc_block = EncoderBlock(sa_layer=sablock, norm0=norm, ffn=df_net, norm1=norm)
            stack = EncoderBlockStack([enc_block] * 3)

            test_input = tf.convert_to_tensor(normal(size=(1, length, d_model)).astype(np.float32))
            mask = to_tensor(([True] * (length - mask_len)) + ([False] * (mask_len)))
            mask = tf.expand_dims(mask, axis=0)
            out = stack(test_input, training=None, mask=mask)

            comp_out_shape = stack.compute_output_shape(test_input.shape)

            self.assertEqual(out.shape, comp_out_shape)

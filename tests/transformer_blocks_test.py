import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from dynastes.blocks.attention_blocks import SelfAttentionBlock1D
from dynastes.blocks.transformer_blocks import PointWiseFeedForwardBlock, EncoderBlock, EncoderBlockStack


def _test_grads(testCase: tf.test.TestCase, func, input):
    _, grads = tf.test.compute_gradient(func, input)
    for grad in grads:
        testCase.assertNotAllClose(grad, np.zeros_like(grad))
        testCase.assertAllInRange(grad, -400., 400)


to_tensor = tf.convert_to_tensor
normal = np.random.normal


class DecoderBlockTest(tf.test.TestCase):
    def test_simple(self):
        d_model = 16
        num_heads = 4
        dff = 32
        max_length = 64
        batch_size = 32
        mask_len = 0
        sablock = SelfAttentionBlock1D(d_model // 4, d_model, num_heads=num_heads,
                                       attention_type='Attention1D',
                                       relative=True,
                                       masked=True,
                                       mask_right=True,
                                       multiquery_attention=True, )
        norm = tfkl.LayerNormalization(epsilon=1e-6)
        df_net = PointWiseFeedForwardBlock(dff=dff, d_model=d_model)
        enc_block = EncoderBlock(sa_layer=sablock, norm0=norm, ffn=df_net, norm1=norm)

        stack = EncoderBlockStack([enc_block] * 3)
        cache = stack.request_cache(batch_size=batch_size, max_length=max_length)

        test_input = tf.convert_to_tensor(normal(size=(batch_size, 1, d_model)).astype(np.float32))
        mask = to_tensor(([True] * (1 - mask_len)) + ([False] * (mask_len)))
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [batch_size, 1])
        enc_out = enc_block(test_input, training=None, mask=mask)
        out = stack(test_input, training=None, mask=mask, cache=cache, decode_loop_step=0)
        comp_out_shape = stack.compute_output_shape(test_input.shape)

        self.assertEqual(out.shape, comp_out_shape)
        self.assertEqual(enc_out.shape, comp_out_shape)

        def inc_encode(out):
            outs = [out]
            for i in range(max_length - 1):
                mask = to_tensor([True])
                mask = tf.expand_dims(mask, axis=0)
                out = stack(out, training=None, mask=mask, cache=cache, decode_loop_step=i + 1)
                outs.append(out)
            outs = tf.concat(outs, axis=1)
            return outs

        ret = inc_encode(out)
        print(ret)

        sanity_check = stack(ret, training=None, mask=tf.cast(tf.ones((batch_size, max_length)), tf.bool))

        print(tf.reduce_max(out - sanity_check))

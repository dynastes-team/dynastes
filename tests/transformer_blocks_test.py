import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util

from dynastes.blocks.attention_blocks import SelfAttentionBlock1D, AttentionBlock1D
from dynastes.blocks.transformer_blocks import PointWiseFeedForwardBlock, EncoderBlock, DecoderBlock, EncoderBlockStack, \
    DecoderBlockStack
from dynastes.layers import BatchNormalization, LayerNormalization
from dynastes.util import cache_context


def _test_grads(testCase: tf.test.TestCase, func, input):
    _, grads = tf.test.compute_gradient(func, input)
    for grad in grads:
        testCase.assertNotAllClose(grad, np.zeros_like(grad))
        testCase.assertAllInRange(grad, -400., 400)


to_tensor = tf.convert_to_tensor
normal = np.random.normal

BatchNormalization
LayerNormalization


class DecoderBlockTest(tf.test.TestCase):

    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with cache_context.CacheContext():
            d_model = 16
            num_heads = 4
            dff = 32
            max_length = 8
            batch_size = 8
            mask_len = 7
            enc_sablock = SelfAttentionBlock1D(d_model // 4, d_model, num_heads=num_heads,
                                               attention_type='Attention1D',
                                               relative=True,
                                               masked=True,
                                               skip_out=False,
                                               mask_right=False,
                                               multiquery_attention=True, )
            enc_norm = LayerNormalization(epsilon=1e-6)
            enc_df_net = PointWiseFeedForwardBlock(dff=dff, d_model=d_model)
            enc_block = EncoderBlock(sa_layer=enc_sablock, norm0=enc_norm, ffn=enc_df_net, norm1=enc_norm)

            stack = EncoderBlockStack([enc_block] * 3)

            enc_input = tf.convert_to_tensor(normal(size=(batch_size, 32, d_model)).astype(np.float32)).numpy()
            enc_mask = to_tensor(([True] * (32 - mask_len)) + ([False] * (mask_len)))
            enc_mask = tf.expand_dims(enc_mask, axis=0)
            enc_mask = tf.tile(enc_mask, [batch_size, 1])
            enc_out = enc_block(enc_input, training=None, mask=enc_mask).numpy()
            encoded_out = stack(enc_input, training=None, mask=enc_mask).numpy()
            comp_out_shape = stack.compute_output_shape(enc_input.shape)

            dec_sablock = SelfAttentionBlock1D(d_model // 4, d_model, num_heads=num_heads,
                                               attention_type='Attention1D',
                                               relative=False,
                                               local=True,
                                               skip_out=False,
                                               masked=True,
                                               mask_right=True,
                                               multiquery_attention=True)
            dec_cablock = AttentionBlock1D(d_model // 4, d_model,
                                           num_heads=num_heads,
                                           attention_type='Attention1D',
                                           relative=False,
                                           masked=True,
                                           skip_out=False,
                                           mask_right=False,
                                           multiquery_attention=True,
                                           cache_kv=False)

            dec_df_net = PointWiseFeedForwardBlock(dff=dff, first_kernel_size=1,
                                                   d_model=d_model)
            dec_blocks = []
            for i in range(5):
                dec_norm_0 = LayerNormalization(epsilon=1e-9)  # , momentum=0.85)
                dec_norm_1 = LayerNormalization(epsilon=1e-9)  # , momentum=0.85)
                dec_norm_2 = LayerNormalization(epsilon=1e-9)  # , momentum=0.85)
                dec_block = DecoderBlock(sa_layer=dec_sablock, ca_layer=dec_cablock, norm0=dec_norm_0, ffn=dec_df_net,
                                         norm1=dec_norm_1, norm2=dec_norm_2)
                dec_blocks.append(dec_block)

            stack = DecoderBlockStack(dec_blocks)
            cache = stack.request_cache(batch_size=batch_size, max_length_sa=max_length, max_length_ca=32)
            dec_input = tf.convert_to_tensor(normal(size=(batch_size, 1, d_model)).astype(np.float32)).numpy()

            for i in range(10):
                inp = tf.convert_to_tensor(normal(size=(batch_size, 32, d_model)).astype(np.float32)).numpy()
                n_mask = np.random.randint(28, 32)
                msk = to_tensor([True] * n_mask + [False] * (32 - n_mask))
                msk = tf.expand_dims(msk, axis=0)
                msk = tf.tile(msk, [batch_size, 1])
                _ = stack((inp, encoded_out), training=True, mask=(msk, enc_mask))

            self.assertEqual(encoded_out.shape, comp_out_shape)
            self.assertEqual(enc_out.shape, comp_out_shape)

            def inc_encode(start, _enc_input, _enc_mask):
                outs = []
                out = start
                for i in range(max_length):
                    mask = to_tensor([True])
                    mask = tf.expand_dims(mask, axis=0)
                    mask = tf.tile(mask, [batch_size, 1])
                    out = stack((out, _enc_input), training=False, mask=(mask, _enc_mask), cache=cache,
                                pad_q_to_kv=True)
                    outs.append(out)
                outs = tf.concat(outs, axis=1)
                return outs

            ret = inc_encode(dec_input, encoded_out, enc_mask)
            check_input = tf.concat([dec_input, ret[:, :-1, :]], axis=1)
            # Call sanity check outside of cache_context to make sure we're getting the same-ish result
        mask = tf.ones(check_input.shape.as_list()[:-1])
        sanity_check = stack((check_input.numpy(), encoded_out), training=False, mask=(
            tf.cast(mask, tf.bool), enc_mask)).numpy()  # , cache=cache, decode_loop_step=0)
        print(tf.reduce_max(ret - sanity_check), tf.reduce_mean(ret - sanity_check))
        # print(dec_df_net.trainable_weights)
        self.assertAllClose(ret, sanity_check)

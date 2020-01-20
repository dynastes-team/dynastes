from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf

from dynastes import activations
from dynastes.blocks import layer_factory
from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.ops import t2t_common
from dynastes.util import cache_context
from dynastes.util.layer_util import call_masked as cm
from dynastes.util.layer_util import compute_mask_if_possible as compm


# A module that only depends on `keras.layers` import these from here.

@tf.keras.utils.register_keras_serializable(package='Dynastes')
class _AttentionBlock1D(DynastesBaseLayer):

    def __init__(self,
                 attention_dim,
                 output_dim,
                 kernel_size=1,
                 attention_type='Attention1D',
                 q_type='Conv1D',
                 k_type=None,
                 v_type=None,
                 num_heads=1,
                 multiquery_attention=False,
                 depth_multiplier=1,
                 strides=1,
                 dilation_rate=1,
                 grouped=False,
                 group_size=1,
                 padding='same',
                 activation=None,
                 use_bias=False,
                 relative=False,
                 local=False,
                 sparse=False,
                 masked=False,
                 dropout_rate=0.,
                 max_relative_position=2,
                 lsh_bucket_length=4,
                 block_length=4,
                 filter_width=2,
                 mask_right=False,
                 add_relative_to_values=False,
                 return_attn_weights=False,
                 cache_kv=False,
                 pad_q_to_kv=False,
                 **kwargs):
        kwargs['supports_caching'] = True
        super(_AttentionBlock1D, self).__init__(**kwargs)
        self.q_type = q_type
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        if k_type is None:
            self.k_type = q_type
        else:
            self.k_type = k_type

        if v_type is None:
            self.v_type = q_type
        else:
            self.v_type = v_type

        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.padding = padding
        self.grouped = grouped
        self.group_size = group_size
        self.multiquery_attention = multiquery_attention
        self.relative = relative
        self.local = local
        self.masked = masked
        self.sparse = sparse
        self.dropout_rate = dropout_rate
        self.max_relative_position = max_relative_position
        self.lsh_bucket_length = lsh_bucket_length
        self.block_length = block_length
        self.filter_width = filter_width
        self.mask_right = mask_right
        self.add_relative_to_values = add_relative_to_values
        self.return_attn_weights = return_attn_weights
        self.supports_masking = True
        self.cache_kv = cache_kv
        self.pad_q_to_kv = pad_q_to_kv
        conv_partial = partial(layer_factory.get_1d_layer, kernel_size=kernel_size,
                               grouped=grouped,
                               group_size=group_size,
                               depth_multiplier=depth_multiplier,
                               padding=padding,
                               activation=activation,
                               use_bias=use_bias,
                               kernel_initializer=self.get_initializer('kernel'),
                               bias_initializer=self.get_initializer('bias'),
                               kernel_regularizer=self.get_regularizer('kernel'),
                               bias_regularizer=self.get_regularizer('bias'),
                               activity_regularizer=None,
                               kernel_constraint=self.get_constraint('kernel'),
                               bias_constraint=self.get_constraint('bias'))
        q_filters = attention_dim
        k_filters = attention_dim
        v_filters = output_dim

        if multiquery_attention:
            k_filters //= num_heads
            v_filters //= num_heads
        self.k_filters = k_filters
        self.v_filters = v_filters

        q_strides = strides
        kv_strides = strides
        kv_dilation_rate = dilation_rate
        attn_strides = 1
        attn_dilation_rate = 1
        if attention_type in ['LocalizedAttentionLayer1D'] and strides != 1:
            kv_strides = 1
            attn_strides = strides
            attn_dilation_rate = dilation_rate

        self.q_layer = conv_partial(type=self.q_type,
                                    filters=q_filters,
                                    strides=q_strides,
                                    dilation_rate=dilation_rate, name='Conv-Q')
        self.k_layer = conv_partial(type=self.k_type,
                                    filters=k_filters,
                                    strides=kv_strides,
                                    dilation_rate=kv_dilation_rate, name='Conv-K')
        self.v_layer = conv_partial(type=self.v_type,
                                    filters=v_filters,
                                    strides=kv_strides,
                                    dilation_rate=kv_dilation_rate, name='Conv-V')

        attention_padding = padding
        self.attention_layer = layer_factory.get_1D_attention_layer(
            type=attention_type,
            strides=attn_strides,
            dilation_rate=attn_dilation_rate,
            num_heads=num_heads,
            padding=attention_padding,
            multiquery_attention=multiquery_attention,
            preshaped_q=True,
            relative=relative,
            local=local,
            masked=masked,
            sparse=sparse,
            dropout_rate=dropout_rate,
            max_relative_position=max_relative_position,
            lsh_bucket_length=lsh_bucket_length,
            block_length=block_length,
            mask_right=mask_right,
            filter_width=filter_width,
            add_relative_to_values=add_relative_to_values,
        )

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            q_mask = compm(self.q_layer, inputs, mask=mask)
            return q_mask
        return mask

    def get_config(self):
        config = {
            'q_type': self.q_type,
            'k_type': self.k_type,
            'v_type': self.v_type,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'multiquery_attention': self.multiquery_attention,
            'attention_dim': self.attention_dim,
            'output_dim': self.output_dim,
            'depth_multiplier': self.depth_multiplier,
            'kernel_size': self.kernel_size,
            'num_heads': self.num_heads,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'padding': self.padding,
            'grouped': self.grouped,
            'group_size': self.group_size,
            'relative': self.relative,
            'local': self.local,
            'masked': self.masked,
            'sparse': self.sparse,
            'dropout_rate': self.dropout_rate,
            'max_relative_position': self.max_relative_position,
            'lsh_bucket_length': self.lsh_bucket_length,
            'block_length': self.block_length,
            'filter_width': self.filter_width,
            'mask_right': self.mask_right,
            'add_relative_to_values': self.add_relative_to_values,
            'cache_kv': self.cache_kv,
            'pad_q_to_kv': self.pad_q_to_kv
        }
        base_config = super(_AttentionBlock1D, self).get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        qs = self.q_layer.compute_output_shape(input_shape)
        ks = self.k_layer.compute_output_shape(input_shape)
        vs = self.v_layer.compute_output_shape(input_shape)
        output_shape = self.attention_layer.compute_output_shape([qs, ks, vs])
        if self.return_attn_weights:
            return output_shape
        return output_shape[0]


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class AttentionBlock1D(_AttentionBlock1D):

    def request_cache(self, batch_size=1, max_length=1):
        return {
        }

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            q_mask = compm(self.q_layer, inputs[0], mask=mask[0])
            return q_mask
        return mask

    def call(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None):
        qx, sx = inputs
        if mask is not None:
            qmask, smask = mask
        else:
            qmask, smask = (None, None)
        q, q_mask = cm(self.q_layer, qx, training=training, mask=qmask)

        # In cross-attention we may compute KV once and perform attention on it in subsequent steps

        def get_kv():
            def do_get_kv():
                k, k_mask = cm(self.k_layer, sx, training=training, mask=smask)
                v, v_mask = cm(self.v_layer, sx, training=training, mask=smask)
                return k, k_mask, v, v_mask

            if self.cache_kv and cache_context.cache_context is not None:
                with cache_context.SubContext(self.name):
                    if 'cached_kv' in cache_context.cache:
                        k, k_mask, v, v_mask = cache_context.cache['cached_kv']
                    else:
                        k, k_mask, v, v_mask = do_get_kv()
                        cache_context.cache['cached_kv'] = k, k_mask, v, v_mask
            else:
                k, k_mask, v, v_mask = do_get_kv()
            return k, k_mask, v, v_mask

        if cache is not None:
            if 'k' not in cache:
                k, k_mask, v, v_mask = get_kv()

                # Update cache
                cache["k"] = k
                cache["v"] = v
                if mask is not None:
                    cache["k_mask"] = k_mask
                    cache["v_mask"] = v_mask
            else:
                k = cache["k"]
                v = cache["v"]
                if mask is not None:
                    k_mask = cache["k_mask"]
                    v_mask = cache["v_mask"]
        else:
            k, k_mask, v, v_mask = get_kv()
        if mask is not None:
            mask = [q_mask, tf.logical_and(k_mask, v_mask)]
        x, weights = self.attention_layer([q, k, v], mask=mask, training=training)

        if self.return_attn_weights:
            return x, weights
        return x


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class SelfAttentionBlock1D(AttentionBlock1D):

    def request_cache(self, batch_size=1, max_length=1):
        return {
            'k': tf.zeros((batch_size, max_length, self.k_filters)),
            'v': tf.zeros((batch_size, max_length, self.v_filters)),
            'k_mask': tf.cast(tf.zeros((batch_size, max_length)), tf.bool),
            'v_mask': tf.cast(tf.zeros((batch_size, max_length)), tf.bool)
        }

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            q_mask = compm(self.q_layer, inputs, mask=mask)
            return q_mask
        return mask

    def call(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None):
        x = inputs
        q, q_mask = cm(self.q_layer, x, training=training, mask=mask)
        k, k_mask = cm(self.k_layer, x, training=training, mask=mask)
        v, v_mask = cm(self.v_layer, x, training=training, mask=mask)
        if cache is not None:
            # Combine cached keys and values with new keys and values.
            if cache["k"] is not None:
                # Update cache
                if decode_loop_step is not None:

                    cache_k_shape = cache["k"].shape.as_list()
                    indices = tf.reshape(
                        tf.one_hot(decode_loop_step, cache_k_shape[1], dtype=k.dtype),
                        [1, cache_k_shape[1], 1])
                    k = cache["k"] + k * indices
                    if mask is not None:
                        indices = tf.reshape(
                            tf.one_hot(decode_loop_step, cache_k_shape[1], dtype=tf.float16),
                            [1, cache_k_shape[1]])
                        k_mask = tf.logical_or(cache["k_mask"], (tf.cast(k_mask, tf.float16) * indices) > 0.)

                    cache_v_shape = cache["v"].shape.as_list()
                    indices = tf.reshape(
                        tf.one_hot(decode_loop_step, cache_v_shape[1], dtype=v.dtype),
                        [1, cache_v_shape[1], 1])
                    v = cache["v"] + v * indices
                    if mask is not None:
                        indices = tf.reshape(
                            tf.one_hot(decode_loop_step, cache_v_shape[1], dtype=tf.float16),
                            [1, cache_v_shape[1]])
                        v_mask = tf.logical_or(cache["v_mask"], (tf.cast(v_mask, tf.float16) * indices) > 0.)
                else:
                    k = tf.concat([tf.cast(cache["k"], k.dtype), k], axis=1)
                    v = tf.concat([tf.cast(cache["v"], v.dtype), v], axis=1)
                    if mask is not None:
                        k_mask = tf.concat([tf.cast(cache["k_mask"], k_mask.dtype), k_mask], axis=1)
                        v_mask = tf.concat([tf.cast(cache["v_mask"], v_mask.dtype), v_mask], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v
            if mask is not None:
                cache["k_mask"] = k_mask
                cache["v_mask"] = v_mask

        if self.pad_q_to_kv:
            q_shape = t2t_common.shape_list(q)
            kv_shape = t2t_common.shape_list(k)
            if q_shape[1] != kv_shape[1]:
                if decode_loop_step is not None:
                    q_prepad = decode_loop_step
                    q_postpad = (kv_shape[1] - q_shape[1]) - decode_loop_step

                else:
                    q_prepad = (kv_shape[1] - q_shape[1])
                    q_postpad = 0
                q = tf.pad(q, paddings=[[0, 0], [q_prepad, q_postpad], [0, 0]])
                if mask is not None:
                    q_mask = tf.pad(q_mask, paddings=[[0, 0], [q_prepad, q_postpad]])

        if mask is not None:
            mask = [q_mask, tf.logical_and(k_mask, v_mask)]
        x, weights = self.attention_layer([q, k, v], mask=mask, training=training)
        if self.pad_q_to_kv:
            if q_shape[1] != kv_shape[1]:
                x_shape = t2t_common.shape_list(x)
                if decode_loop_step is not None:
                    x = tf.slice(x, [0, q_prepad, 0], [x_shape[0], 1, x_shape[2]])
                else:
                    x = tf.slice(x, [0, q_prepad, 0], [x_shape[0], q_shape[1], x_shape[2]])
        if self.return_attn_weights:
            return x, weights
        return x

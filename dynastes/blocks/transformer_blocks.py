from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.keras.layers as tfkl

from dynastes import activations
from dynastes.blocks import layer_factory
from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.util import cache_context
from dynastes.util.layer_util import call_masked as cm


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class PointWiseFeedForwardBlock(DynastesBaseLayer):

    def __init__(self,
                 dff,
                 d_model,
                 kernel_size=1,
                 ff_type='Dense',
                 d_type='Dense',
                 depth_multiplier=1,
                 strides=1,
                 dilation_rate=1,
                 grouped=False,
                 group_size=1,
                 padding='same',
                 activation=None,
                 use_bias=False,
                 dropout_rate=0.,
                 **kwargs):
        super(PointWiseFeedForwardBlock, self).__init__(**kwargs)
        self.dff = dff
        self.d_model = d_model
        self.ff_type = ff_type
        self.d_type = d_type
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.supports_masking = True

        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.padding = padding
        self.grouped = grouped
        self.group_size = group_size
        self.dropout_rate = dropout_rate

        conv_partial = partial(layer_factory.get_1d_layer, kernel_size=kernel_size,
                               grouped=grouped,
                               group_size=group_size,
                               depth_multiplier=depth_multiplier,
                               padding=padding,
                               use_bias=use_bias,
                               strides=strides,
                               dilation_rate=dilation_rate,
                               kernel_initializer=self.get_initializer('kernel'),
                               bias_initializer=self.get_initializer('bias'),
                               kernel_regularizer=self.get_regularizer('kernel'),
                               bias_regularizer=self.get_regularizer('bias'),
                               activity_regularizer=None,
                               kernel_constraint=self.get_constraint('kernel'),
                               bias_constraint=self.get_constraint('bias'))

        self.dff_layer = conv_partial(
            type=ff_type,
            filters=dff,
            activation=self.activation)
        self.out_layer = conv_partial(
            type=d_type,
            filters=d_model,
            activation=None)

    def call_masked(self, inputs, training=None, mask=None):
        x, x_mask = cm(self.dff_layer, inputs, training=training, mask=mask)
        x, x_mask = cm(self.out_layer, x, training=training, mask=x_mask)
        return x, x_mask

    def call(self, inputs, training=None, mask=None):
        x, x_mask = cm(self.dff_layer, inputs, training=training, mask=mask)
        x = self.out_layer(x, training=training, mask=x_mask)
        return x

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.dff_layer.supports_masking:
                mask = self.dff_layer.compute_mask(inputs, mask=mask)
            if self.out_layer.supports_masking:
                mask = self.out_layer.compute_mask(inputs, mask=mask)
        return mask

    def compute_output_shape(self, input_shape):
        shape = self.dff_layer.compute_output_shape(input_shape)
        shape = self.out_layer.compute_output_shape(shape)
        return shape

    def get_config(self):
        config = {
            'dff': self.dff,
            'd_model': self.d_model,
            'ff_type': self.ff_type,
            'd_type': self.d_type,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'depth_multiplier': self.depth_multiplier,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'padding': self.padding,
            'grouped': self.grouped,
            'group_size': self.group_size,
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(PointWiseFeedForwardBlock, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class EncoderBlock(tfkl.Layer):

    def __init__(self,
                 sa_layer: tfkl.Layer,
                 norm0: tfkl.Layer,
                 ffn: tfkl.Layer,
                 norm1: tfkl.Layer,
                 mha_skip_adapt: tfkl.Layer = tfkl.Activation('linear'),
                 ffn_skip_adapt: tfkl.Layer = tfkl.Activation('linear'),
                 **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.sa_layer = sa_layer
        self.norm0 = norm0
        self.ffn = ffn
        self.norm1 = norm1
        self.mha_skip_adapt = mha_skip_adapt
        self.ffn_skip_adapt = ffn_skip_adapt

    def request_cache(self, batch_size=1, max_length=1):
        try:
            return self.sa_layer.request_cache(batch_size=batch_size, max_length=max_length)
        except:
            return None

    def call_masked(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None):
        with cache_context.SubContext(self.name):
            x = inputs
            x, x_mask = cm(self.sa_layer, x, training=training, mask=mask, cache=cache,
                           decode_loop_step=decode_loop_step)
            res, res_mask = cm(self.mha_skip_adapt, inputs, training=training, mask=mask)
            if x_mask is None:
                n_mask = res_mask
            elif res_mask is None:
                n_mask = x_mask
            else:
                n_mask = tf.math.logical_and(x_mask, res_mask)
            x, x_mask = cm(self.norm0, x + res, training=training, mask=n_mask)

            f, f_mask = cm(self.ffn, x, training=training, mask=x_mask)
            res, res_mask = cm(self.ffn_skip_adapt, x, training=training, mask=x_mask)
            if f_mask is None:
                n_mask = f_mask
            elif res_mask is None:
                n_mask = x_mask
            else:
                n_mask = tf.math.logical_and(f_mask, res_mask)
            x, mask = cm(self.norm1, f + res, training=training, mask=n_mask)
            return x, mask

    def call(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None):
        with cache_context.SubContext(self.name):
            x = inputs
            x, x_mask = cm(self.sa_layer, x, training=training, mask=mask, cache=cache,
                           decode_loop_step=decode_loop_step)
            res, res_mask = cm(self.mha_skip_adapt, inputs, training=training, mask=mask)
            if x_mask is None:
                n_mask = res_mask
            elif res_mask is None:
                n_mask = x_mask
            else:
                n_mask = tf.math.logical_and(x_mask, res_mask)
            x, x_mask = cm(self.norm0, x + res, training=training, mask=n_mask)

            f, f_mask = cm(self.ffn, x, training=training, mask=x_mask)
            res, res_mask = cm(self.ffn_skip_adapt, x, training=training, mask=x_mask)
            if f_mask is None:
                n_mask = f_mask
            elif res_mask is None:
                n_mask = x_mask
            else:
                n_mask = tf.math.logical_and(f_mask, res_mask)
            x, _ = cm(self.norm1, f + res, training=training, mask=n_mask)
            return x

    def compute_mask(self, inputs, mask=None):
        with cache_context.SubContext(self.name):
            x = inputs
            if self.sa_layer.supports_masking:
                x_mask = self.sa_layer.compute_mask(x, mask=mask)
            else:
                x_mask = mask
            if self.mha_skip_adapt.supports_masking:
                res_mask = self.mha_skip_adapt.compute_mask(inputs, mask=mask)
            else:
                res_mask = mask
            if x_mask is None:
                n_mask = res_mask
            elif res_mask is None:
                n_mask = x_mask
            else:
                n_mask = tf.math.logical_and(x_mask, res_mask)
            if self.norm0.supports_masking:
                x_mask = self.norm0.compute_mask(x, mask=n_mask)
            else:
                x_mask = n_mask
            if self.ffn.supports_masking:
                f_mask = self.ffn.compute_mask(x, mask=x_mask)
            else:
                f_mask = x_mask
            if self.ffn_skip_adapt.supports_masking:
                res_mask = self.ffn_skip_adapt.compute_mask(x, mask=x_mask)
            else:
                res_mask = x_mask
            if f_mask is None:
                n_mask = f_mask
            elif res_mask is None:
                n_mask = x_mask
            else:
                n_mask = tf.math.logical_and(f_mask, res_mask)
            if self.norm1.supports_masking:
                mask = self.norm1.compute_mask(x, mask=n_mask)
            else:
                mask = n_mask
            return mask

    def compute_output_shape(self, input_shape):
        s = input_shape
        s = self.mha_skip_adapt.compute_output_shape(s)
        s = self.ffn_skip_adapt.compute_output_shape(s)
        return s


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class EncoderBlockStack(tfkl.Layer):

    def __init__(self,
                 blocks,
                 **kwargs):
        super(EncoderBlockStack, self).__init__(**kwargs)
        self.blocks = blocks

    def request_cache(self, batch_size=1, max_length=1):
        cache = {}
        for i, block in enumerate(self.blocks):
            try:
                block_cache = block.request_cache(batch_size=batch_size, max_length=max_length)
            except:
                block_cache = None
            cache[i] = block_cache
        return cache

    def call_masked(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None, **kwargs):
        with cache_context.SubContext(self.name):
            x = inputs
            for i, block in enumerate(self.blocks):
                if cache is not None:
                    block_cache = cache[i]
                else:
                    block_cache = None
                x, mask = cm(block, x, training=training, mask=mask, cache=block_cache,
                             decode_loop_step=decode_loop_step,
                             **kwargs)
            return x, mask

    def call(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None, **kwargs):
        with cache_context.SubContext(self.name):
            x = inputs
            for i, block in enumerate(self.blocks):
                if cache is not None:
                    block_cache = cache[i]
                else:
                    block_cache = None
                x = block(x, training=training, mask=mask, cache=block_cache, decode_loop_step=decode_loop_step,
                          **kwargs)
            return x

    def compute_mask(self, inputs, mask=None):
        with cache_context.SubContext(self.name):
            x = inputs
            for block in self.blocks:
                if block.supports_masking:
                    mask = block.compute_mask(x, mask=mask)
            return mask

    def compute_output_shape(self, input_shape):
        s = input_shape
        for block in self.blocks:
            s = block.compute_output_shape(s)
        return s


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DecoderBlock(tfkl.Layer):

    def __init__(self,
                 sa_layer: tfkl.Layer,
                 norm0: tfkl.Layer,
                 ca_layer: tfkl.Layer,
                 norm1: tfkl.Layer,
                 ffn: tfkl.Layer,
                 norm2: tfkl.Layer,
                 mha_skip_adapt: tfkl.Layer = tfkl.Activation('linear'),
                 ffn_skip_adapt: tfkl.Layer = tfkl.Activation('linear'),
                 **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.sa_layer = sa_layer
        self.norm0 = norm0
        self.ca_layer = ca_layer
        self.norm1 = norm1
        self.ffn = ffn
        self.norm2 = norm2
        self.mha_skip_adapt = mha_skip_adapt
        self.ffn_skip_adapt = ffn_skip_adapt

    def request_cache(self, batch_size=1, max_length_sa=1, max_length_ca=1):
        try:
            return {'sa': self.sa_layer.request_cache(batch_size=batch_size, max_length=max_length_sa),
                    'ca': self.ca_layer.request_cache(batch_size=batch_size, max_length=max_length_ca)}
        except:
            return None

    def _call(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None):
        x, enc_in = inputs
        _x = x
        if mask is not None:
            x_mask, enc_mask = mask
        else:
            assert False
            x_mask = None
            enc_mask = None
        _x_mask = x_mask
        ## Self-attention
        if cache is not None:
            sa_cache = cache['sa']
        else:
            sa_cache = None
        x, x_mask = cm(self.sa_layer, x, training=training, mask=x_mask, cache=sa_cache,
                       decode_loop_step=decode_loop_step)
        res, res_mask = cm(self.mha_skip_adapt, _x, training=training, mask=_x_mask)
        if x_mask is None:
            n_mask = res_mask
        elif res_mask is None:
            n_mask = x_mask
        else:
            n_mask = tf.math.logical_and(x_mask, res_mask)
        x, x_mask = cm(self.norm0, x + res, training=training, mask=n_mask)
        _x = x
        ## Attend to encoding
        if mask is not None:
            ca_mask = x_mask, enc_mask
        else:
            assert False
            ca_mask = None
        if cache is not None:
            ca_cache = cache['ca']
        else:
            ca_cache = None
        x, x_mask = cm(self.ca_layer, (x, enc_in), training=training, mask=ca_mask, cache=ca_cache)

        res, res_mask = cm(self.mha_skip_adapt, _x, training=training, mask=x_mask)
        if x_mask is None:
            n_mask = res_mask
        elif res_mask is None:
            n_mask = x_mask
        else:
            n_mask = tf.math.logical_and(x_mask, res_mask)

        x, x_mask = cm(self.norm1, x + res, training=training, mask=n_mask)
        _x = x
        ## FF-net
        f, f_mask = cm(self.ffn, x, training=training, mask=x_mask)
        res, res_mask = cm(self.ffn_skip_adapt, _x, training=training, mask=x_mask)
        if f_mask is None:
            n_mask = f_mask
        elif res_mask is None:
            n_mask = x_mask
        else:
            n_mask = tf.math.logical_and(f_mask, res_mask)
        x, mask = cm(self.norm2, f + res, training=training, mask=n_mask)

        return x, mask

    def call_masked(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None):
        return self._call(inputs, training=training, mask=mask, cache=cache, decode_loop_step=decode_loop_step)

    def call(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None):
        x, _ = self._call(inputs, training=training, mask=mask, cache=cache, decode_loop_step=decode_loop_step)
        return x

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        s = input_shape
        s = self.mha_skip_adapt.compute_output_shape(s)
        s = self.ffn_skip_adapt.compute_output_shape(s)
        return s


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DecoderBlockStack(tfkl.Layer):

    def __init__(self,
                 blocks,
                 **kwargs):
        super(DecoderBlockStack, self).__init__(**kwargs)
        self.blocks = blocks

    def request_cache(self, batch_size=1, max_length_ca=1, max_length_sa=1):
        cache = {}
        for i, block in enumerate(self.blocks):
            try:
                block_cache = block.request_cache(batch_size=batch_size, max_length_ca=max_length_ca,
                                                  max_length_sa=max_length_sa)
            except:
                block_cache = None
            cache[i] = block_cache
        return cache

    def call_masked(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None, **kwargs):
        x = inputs
        for i, block in enumerate(self.blocks):
            if cache is not None:
                block_cache = cache[i]
            else:
                block_cache = None
            x, mask = cm(block, x, training=training, mask=mask, cache=block_cache, decode_loop_step=decode_loop_step,
                         **kwargs)
        return x, mask

    def call(self, inputs, training=None, mask=None, cache=None, decode_loop_step=None, **kwargs):
        x, enc = inputs
        for i, block in enumerate(self.blocks):
            if cache is not None:
                block_cache = cache[i]
            else:
                block_cache = None
            x = block((x, enc), training=training, mask=mask, cache=block_cache, decode_loop_step=decode_loop_step,
                      **kwargs)
        return x

    def compute_mask(self, inputs, mask=None):
        x = inputs
        for block in self.blocks:
            if block.supports_masking:
                mask = block.compute_mask(x, mask=mask)
        return mask

    def compute_output_shape(self, input_shape):
        s = input_shape
        for block in self.blocks:
            s = block.compute_output_shape(s)
        return s

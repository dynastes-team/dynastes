from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras import activations

from dynastes.blocks import layer_factory
from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.util.layer_util import call_masked as cm


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
            activation=None)
        self.out_layer = conv_partial(
            type=d_type,
            filters=d_model,
            activation=self.activation)

    def call_masked(self, inputs, training=None, mask=None):
        x, x_mask = cm(self.dff_layer, inputs, training=training, mask=mask)
        x, x_mask = cm(self.out_layer, x, training=training, mask=x_mask)
        return x, x_mask

    def call(self, inputs, training=None, mask=None):
        return self.call_masked(inputs, training=training, mask=mask)[0]

    def compute_mask(self, inputs, mask=None):
        return self.call_masked(inputs, training=None, mask=mask)[1]

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

    def call_masked(self, inputs, training=None, mask=None):
        x = inputs
        x, x_mask = cm(self.sa_layer, x, training=training, mask=mask)
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
        x, mask = cm(self.norm0, f + res, training=training, mask=n_mask)
        return x, mask

    def call(self, inputs, training=None, mask=None):
        return self.call_masked(inputs, training=training, mask=mask)[0]

    def compute_mask(self, inputs, mask=None):
        return self.call_masked(inputs, mask=mask)[1]

    def compute_output_shape(self, input_shape):
        s = input_shape
        s = self.mha_skip_adapt.compute_output_shape(s)
        s = self.ffn_skip_adapt.compute_output_shape(s)
        return s


class EncoderBlockStack(tfkl.Layer):

    def __init__(self,
                 blocks,
                 **kwargs):
        super(EncoderBlockStack, self).__init__(**kwargs)
        self.blocks = blocks

    def call_masked(self, inputs, training=None, mask=None, **kwargs):
        x = inputs
        for block in self.blocks:
            x, mask = cm(block, x, training=training, mask=mask, **kwargs)
        return x, mask

    def call(self, inputs, training=None, mask=None):
        return self.call_masked(inputs, training=training, mask=mask)[0]

    def compute_mask(self, inputs, mask=None):
        return self.call_masked(inputs, mask=mask)[1]

    def compute_output_shape(self, input_shape):
        s = input_shape
        for block in self.blocks:
            s = block.compute_output_shape(s)
        return s
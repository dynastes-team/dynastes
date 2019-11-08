from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import nn

from dynastes import layers as vqkl
from dynastes.blocks import layer_factory
from dynastes.layers import ActivatedKernelBiasBaseLayer


class LocalizedSelfAttentionBlock1D(ActivatedKernelBiasBaseLayer):

    def __init__(self,
                 attention_dim,
                 output_dim,
                 kernel_size,
                 q_type='Conv1D',
                 k_type='Conv1D',
                 v_type='Conv1D',
                 num_heads=1,
                 depth_multiplier=1,
                 strides=1,
                 dilation_rate=1,
                 grouped=False,
                 group_size=1,
                 padding='same',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='he_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(LocalizedSelfAttentionBlock1D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        conv_partial = partial(layer_factory.get_1d_layer(kernel_size=kernel_size,
                                                          grouped=grouped,
                                                          group_size=group_size,
                                                          padding=padding,
                                                          activation=None,
                                                          use_bias=True,
                                                          kernel_initializer=self.kernel_initializer,
                                                          bias_initializer=self.bias_initializer,
                                                          kernel_regularizer=self.kernel_regularizer,
                                                          bias_regularizer=self.bias_regularizer,
                                                          activity_regularizer=None,
                                                          kernel_constraint=self.kernel_constraint,
                                                          bias_constraint=self.bias_constraint))
        self.q_layer = conv_partial(type=q_type,
                                    output_dim=attention_dim,
                                    output_mul=depth_multiplier,
                                    stride=strides,
                                    dilation=dilation_rate, name='Conv-Q')
        self.k_layer = conv_partial(type=k_type,
                                    output_dim=attention_dim,
                                    output_mul=depth_multiplier,
                                    stride=1,
                                    dilation=1, name='Conv-K')
        self.v_layer = conv_partial(type=v_type,
                                    output_dim=output_dim,
                                    output_mul=depth_multiplier,
                                    stride=1,
                                    dilation=1, name='Conv-K')

        attention_padding = padding
        self.attention_layer = vqkl.LocalizedAttentionLayer1D(strides=strides,
                                                              dilation_rate=dilation_rate,
                                                              num_heads=num_heads,
                                                              padding=attention_padding,
                                                              preshaped_q=True)

    def build(self, input_shape):
        self.build_bias(self.output_dim)

    def call(self, inputs, **kwargs):

        x = inputs
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)

        x = self.attention_layer(q=q, k=k, v=v)

        return super(LocalizedSelfAttentionBlock1D, self).call(x, **kwargs)

    def get_config(self):
        config = {
            'attention_dim': self.attention_dim,
            'output_dim': self.output_dim,
            'kernel_size': self.kernel_size,
            'num_heads': self.num_heads,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'padding': self.padding,
        }
        base_config = super(LocalizedSelfAttentionBlock1D, self).get_config()
        return {**base_config, **config}

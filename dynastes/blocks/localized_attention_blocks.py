from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from tensorflow.python.keras import activations

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
                 multiquery_attention=False,
                 depth_multiplier=1,
                 strides=1,
                 dilation_rate=1,
                 grouped=False,
                 group_size=1,
                 padding='same',
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(LocalizedSelfAttentionBlock1D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
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
        self.multiquery_attention = multiquery_attention

        conv_partial = partial(layer_factory.get_1d_layer(kernel_size=kernel_size,
                                                          grouped=grouped,
                                                          group_size=group_size,
                                                          depth_multiplier=depth_multiplier,
                                                          padding=padding,
                                                          activation=None,
                                                          use_bias=True,
                                                          kernel_initializer=self.get_initializer('kernel'),
                                                          bias_initializer=self.get_initializer('bias'),
                                                          kernel_regularizer=self.get_regularizer('kernel'),
                                                          bias_regularizer=self.get_regularizer('bias'),
                                                          activity_regularizer=None,
                                                          kernel_constraint=self.get_constraint('kernel'),
                                                          bias_constraint=self.get_constraint('bias')))
        q_filters = attention_dim
        k_filters = attention_dim
        v_filters = output_dim

        if multiquery_attention:
            k_filters /= num_heads
            v_filters /= num_heads

        self.q_layer = conv_partial(type=q_type,
                                    filters=q_filters,
                                    stride=strides,
                                    dilation=dilation_rate, name='Conv-Q')
        self.k_layer = conv_partial(type=k_type,
                                    filters=k_filters,
                                    stride=1,
                                    dilation=1, name='Conv-K')
        self.v_layer = conv_partial(type=v_type,
                                    filters=v_filters,
                                    stride=1,
                                    dilation=1, name='Conv-V')

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

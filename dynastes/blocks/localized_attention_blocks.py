from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from tensorflow.keras import layers as tfkl

from dynastes import layers as vqkl
from dynastes.blocks import layer_factory


class LocalizedSelfAttentionBlock1D(tfkl.Layer):

    def __init__(self,
                 attention_dim,
                 output_dim,
                 context_size,
                 q_type='Conv1D',
                 k_type='Conv1D',
                 v_type='Conv1D',
                 num_heads=1,
                 output_mul=1,
                 stride=1,
                 dilation=1,
                 grouped=False,
                 group_size=1,
                 padding='same',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(LocalizedSelfAttentionBlock1D, self).__init__(**kwargs)
        conv_partial = partial(layer_factory.get_1d_layer(context_size=context_size,
                                                          grouped=grouped,
                                                          group_size=group_size,
                                                          padding=padding,
                                                          activation=None,
                                                          use_bias=True,
                                                          kernel_initializer=kernel_initializer,
                                                          bias_initializer=bias_initializer,
                                                          kernel_regularizer=kernel_regularizer,
                                                          bias_regularizer=bias_regularizer,
                                                          activity_regularizer=activity_regularizer,
                                                          kernel_constraint=kernel_constraint,
                                                          bias_constraint=bias_constraint))
        self.q_layer = conv_partial(type=q_type,
                                    output_dim=attention_dim,
                                    output_mul=output_mul,
                                    stride=stride,
                                    dilation=dilation, name='Conv-Q')
        self.k_layer = conv_partial(type=k_type,
                                    output_dim=attention_dim,
                                    output_mul=output_mul,
                                    stride=1,
                                    dilation=1, name='Conv-K')
        self.v_layer = conv_partial(type=v_type,
                                    output_dim=output_dim,
                                    output_mul=output_mul,
                                    stride=1,
                                    dilation=1, name='Conv-K')

        attention_padding = padding
        if attention_padding == 'causal':
            attention_padding = 'same'
        self.attention_layer = vqkl.attention_layers.LocalizedAttentionLayer1D(stride=stride,
                                                                               dilation=dilation,
                                                                               num_heads=num_heads,
                                                                               padding=attention_padding,
                                                                               preshaped_q=True)

    def call(self, inputs, **kwargs):
        x = inputs
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)

        return self.attention_layer(q=q, k=k, v=v)

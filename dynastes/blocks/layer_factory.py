from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import layers as tfkl

from dynastes import layers as vqkl


def get_1d_layer(type,
                 output_dim,
                 output_mul,
                 context_size,
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
    if type.lower() == 'TimeDelayLayer1D'.lower():
        if padding == 'causal':
            padding = 'same'
        return vqkl.time_delay_layers.TimeDelayLayer1D(output_dim=output_dim,
                                                       context_size=context_size,
                                                       stride=stride,
                                                       dilation=dilation,
                                                       padding=padding,
                                                       activation=activation,
                                                       use_bias=use_bias,
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer,
                                                       kernel_regularizer=kernel_regularizer,
                                                       bias_regularizer=bias_regularizer,
                                                       activity_regularizer=activity_regularizer,
                                                       kernel_constraint=kernel_constraint,
                                                       bias_constraint=bias_constraint, **kwargs)
    elif type.lower() == 'DepthGroupwiseTimeDelayLayer1D'.lower():
        if padding == 'causal':
            padding = 'same'
        return vqkl.time_delay_layers.DepthGroupwiseTimeDelayLayer1D(output_mul=output_mul,
                                                                     context_size=context_size,
                                                                     stride=stride,
                                                                     dilation=dilation,
                                                                     padding=padding,
                                                                     activation=activation,
                                                                     use_bias=use_bias,
                                                                     grouped=grouped,
                                                                     group_size=group_size,
                                                                     kernel_initializer=kernel_initializer,
                                                                     bias_initializer=bias_initializer,
                                                                     kernel_regularizer=kernel_regularizer,
                                                                     bias_regularizer=bias_regularizer,
                                                                     activity_regularizer=activity_regularizer,
                                                                     kernel_constraint=kernel_constraint,
                                                                     bias_constraint=bias_constraint, **kwargs)
    elif type.lower() in ['Convolution1D'.lower(), 'Conv1D'.lower()]:
        return tfkl.Convolution1D(filters=output_dim,
                                  kernel_size=context_size,
                                  strides=stride,
                                  dilation_rate=dilation,
                                  padding=padding,
                                  activation=activation,
                                  use_bias=use_bias,
                                  grouped=grouped,
                                  group_size=group_size,
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  activity_regularizer=activity_regularizer,
                                  kernel_constraint=kernel_constraint,
                                  bias_constraint=bias_constraint, **kwargs)
    elif type.lower() in ['SeparableConv1D'.lower(), 'SeparableConvolution1D'.lower()]:
        return tfkl.SeparableConvolution1D(filters=output_dim,
                                           kernel_size=context_size,
                                           strides=stride,
                                           dilation_rate=dilation,
                                           padding=padding,
                                           activation=activation,
                                           use_bias=use_bias,
                                           grouped=grouped,
                                           group_size=group_size,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=kernel_regularizer,
                                           bias_regularizer=bias_regularizer,
                                           activity_regularizer=activity_regularizer,
                                           kernel_constraint=kernel_constraint,
                                           bias_constraint=bias_constraint, **kwargs)

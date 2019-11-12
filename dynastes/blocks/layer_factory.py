from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import layers as tfkl

from dynastes import layers as vqkl


def get_1d_layer(type,
                 filters,
                 depth_multiplier,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
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
        return vqkl.TimeDelayLayer1D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     dilation_rate=dilation_rate,
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
        return vqkl.DepthGroupwiseTimeDelayLayer1D(depth_multiplier=depth_multiplier,
                                                   kernel_size=kernel_size,
                                                   strides=strides,
                                                   dilation_rate=dilation_rate,
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
        return tfkl.Convolution1D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  dilation_rate=dilation_rate,
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
        return tfkl.SeparableConvolution1D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           dilation_rate=dilation_rate,
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

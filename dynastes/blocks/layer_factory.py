from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras.layers as tfkl
from tensorflow.keras import Sequential

from dynastes import activations
from dynastes import layers
from dynastes.probability.pseudoblocksparse_bijectors import BlockSparseStridedRoll1D


def get_1d_layer(type,
                 **kwargs):
    types = type.split('|')
    if len(types) == 1:
        return _get_1d_layer(type, **kwargs)
    else:
        return Sequential([_get_1d_layer(t, **kwargs) for t in types])


def _get_1d_layer(type,
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
                  kernel_normalizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  separable_prepointwise=False,
                  separable_prepointwise_depth='min',
                  **kwargs):
    if type.lower() == 'TimeDelayLayer1D'.lower():
        return layers.TimeDelayLayer1D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       dilation_rate=dilation_rate,
                                       padding=padding,
                                       activation=activation,
                                       use_bias=use_bias,
                                       kernel_initializer=kernel_initializer,
                                       kernel_normalizer=kernel_normalizer,
                                       bias_initializer=bias_initializer,
                                       kernel_regularizer=kernel_regularizer,
                                       bias_regularizer=bias_regularizer,
                                       activity_regularizer=activity_regularizer,
                                       kernel_constraint=kernel_constraint,
                                       bias_constraint=bias_constraint, **kwargs)
    elif type.lower() == 'DepthGroupwiseTimeDelayLayer1D'.lower():
        return layers.DepthGroupwiseTimeDelayLayer1D(depth_multiplier=depth_multiplier,
                                                     kernel_size=kernel_size,
                                                     strides=strides,
                                                     dilation_rate=dilation_rate,
                                                     padding=padding,
                                                     grouped=grouped,
                                                     group_size=group_size,
                                                     activation=activation,
                                                     use_bias=use_bias,
                                                     kernel_normalizer=kernel_normalizer,
                                                     kernel_initializer=kernel_initializer,
                                                     bias_initializer=bias_initializer,
                                                     kernel_regularizer=kernel_regularizer,
                                                     bias_regularizer=bias_regularizer,
                                                     activity_regularizer=activity_regularizer,
                                                     kernel_constraint=kernel_constraint,
                                                     bias_constraint=bias_constraint, **kwargs)
    elif type.lower() in ['Convolution1D'.lower(), 'Conv1D'.lower()]:
        return layers.DynastesConv1D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     dilation_rate=dilation_rate,
                                     padding=padding,
                                     activation=activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     kernel_normalizer=kernel_normalizer,
                                     bias_initializer=bias_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     activity_regularizer=activity_regularizer,
                                     kernel_constraint=kernel_constraint,
                                     bias_constraint=bias_constraint, **kwargs)
    elif type.lower() in ['SeparableConvolution1D'.lower(), 'SeparableConv1D'.lower()]:
        return layers.DynastesSeparableConv1D(filters=filters,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              dilation_rate=dilation_rate,
                                              padding=padding,
                                              activation=activation,
                                              use_bias=use_bias,
                                              prepointwise_kernel_initializer=kernel_initializer,
                                              prepointwise_kernel_normalizer=kernel_normalizer,
                                              prepointwise_bias_initializer=bias_initializer,
                                              prepointwise_kernel_regularizer=kernel_regularizer,
                                              prepointwise_bias_regularizer=bias_regularizer,
                                              prepointwise_activity_regularizer=activity_regularizer,
                                              prepointwise_kernel_constraint=kernel_constraint,
                                              prepointwise_bias_constraint=bias_constraint,
                                              pointwise_kernel_initializer=kernel_initializer,
                                              pointwise_kernel_normalizer=kernel_normalizer,
                                              pointwise_bias_initializer=bias_initializer,
                                              pointwise_kernel_regularizer=kernel_regularizer,
                                              pointwise_bias_regularizer=bias_regularizer,
                                              pointwise_activity_regularizer=activity_regularizer,
                                              pointwise_kernel_constraint=kernel_constraint,
                                              pointwise_bias_constraint=bias_constraint,
                                              depthwise_kernel_initializer=kernel_initializer,
                                              depthwise_kernel_normalizer=kernel_normalizer,
                                              depthwise_kernel_regularizer=kernel_regularizer,
                                              depthwise_kernel_constraint=kernel_constraint,
                                              prepointwise=separable_prepointwise,
                                              prepointwise_depth=separable_prepointwise_depth,
                                              **kwargs)
    elif type.lower() in ['DepthwiseConv1D'.lower(), 'DepthwiseConvolution1D'.lower()]:
        return layers.DynastesDepthwiseConv1D(kernel_size=kernel_size,
                                              strides=strides,
                                              dilation_rate=dilation_rate,
                                              padding=padding,
                                              activation=activation,
                                              use_bias=use_bias,
                                              kernel_initializer=kernel_initializer,
                                              kernel_normalizer=kernel_normalizer,
                                              bias_initializer=bias_initializer,
                                              kernel_regularizer=kernel_regularizer,
                                              bias_regularizer=bias_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint,
                                              bias_constraint=bias_constraint, **kwargs)
    elif type.lower() in ['Dense'.lower()]:
        return layers.DynastesDense(units=filters,
                                    activation=activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    kernel_normalizer=kernel_normalizer,
                                    bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_constraint=bias_constraint, **kwargs)
    elif type.lower() in ['relu', 'mish', 'swish', 'None', 'sigmoid']:
        return tfkl.Activation(activations.get(type))
    else:
        return tfkl.Activation('linear')


def get_1D_attention_layer(type,
                           strides,
                           dilation_rate,
                           num_heads,
                           padding,
                           multiquery_attention,
                           self_attention=True,
                           preshaped_q=True,
                           relative=False,
                           local=False,
                           sparse=False,
                           masked=False,
                           dropout_rate=0.,
                           max_relative_position=2,
                           blocksparse_bijector=BlockSparseStridedRoll1D,
                           lsh_bucket_length=4,
                           block_length=None,
                           filter_width=None,
                           mask_right=False,
                           add_relative_to_values=False,
                           heads_share_relative_embeddings=False,
                           scaled=False):
    if type.lower() == 'LocalizedAttentionLayer1D'.lower():
        return layers.LocalizedAttentionLayer1D(strides=strides,
                                                dilation_rate=dilation_rate,
                                                num_heads=num_heads,
                                                padding=padding,
                                                preshaped_q=preshaped_q)
    elif type.lower() == 'Attention1D'.lower():
        return layers.Attention1D(num_heads=num_heads,
                                  multiquery_attention=multiquery_attention,
                                  self_attention=self_attention,
                                  relative=relative,
                                  masked=masked,
                                  sparse=sparse,
                                  local=local,
                                  dropout_rate=dropout_rate,
                                  max_relative_position=max_relative_position,
                                  lsh_bucket_length=lsh_bucket_length,
                                  block_length=block_length,
                                  filter_width=filter_width,
                                  mask_right=mask_right,
                                  add_relative_to_values=add_relative_to_values,
                                  heads_share_relative_embeddings=heads_share_relative_embeddings,
                                  scaled=scaled)
    elif type.lower() == 'PseudoBlockSparseAttention1D'.lower():
        return layers.PseudoBlockSparseAttention1D(num_heads=num_heads,
                                                   blocksparse_bijector=blocksparse_bijector(block_size=block_length),
                                                   multiquery_attention=multiquery_attention,
                                                   dropout_rate=dropout_rate,
                                                   block_size=block_length,
                                                   mask_right=mask_right)

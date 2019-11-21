from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from tensorflow.python.keras import activations

from dynastes.blocks import layer_factory
from dynastes.layers import ActivatedKernelBiasBaseLayer


class SelfAttentionBlock1D(ActivatedKernelBiasBaseLayer):

    def __init__(self,
                 attention_dim,
                 output_dim,
                 kernel_size,
                 attention_type='LocalizedAttentionLayer1D',
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
                 max_relative_position=None,
                 lsh_bucket_length=4,
                 block_length=128,
                 filter_width=100,
                 mask_right=False,
                 add_relative_to_values=False,
                 **kwargs):
        super(SelfAttentionBlock1D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            **kwargs)
        self.q_type = q_type

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
        self.attention_layer = layer_factory.get_1D_attention_layer(
            type=attention_type,
            strides=strides,
            dilation_rate=dilation_rate,
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

    def build(self, input_shape):
        self.build_bias(self.output_dim)

    def call(self, inputs, training=None, mask=None, **kwargs):
        x = inputs
        q = self.q_layer(x, training=training)
        k = self.k_layer(x, training=training)
        v = self.v_layer(x, training=training)

        if mask is not None:
            mask = [mask, mask, mask]
        x, weights = self.attention_layer([q,k,v], mask=mask, training=training)

        return super(SelfAttentionBlock1D, self).call(x, **kwargs), weights

    def get_config(self):
        config = {
            'q_type': self.q_type,
            'k_type': self.k_type,
            'v_type': self.v_type,
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
        }
        base_config = super(SelfAttentionBlock1D, self).get_config()
        return {**base_config, **config}

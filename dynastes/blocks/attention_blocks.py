from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import activations
# A module that only depends on `keras.layers` import these from here.
from tensorflow.python.util import nest

from dynastes.blocks import layer_factory
from dynastes.layers import ActivatedKernelBiasBaseLayer


class SelfAttentionBlock1D(ActivatedKernelBiasBaseLayer):

    def __init__(self,
                 attention_dim,
                 output_dim,
                 kernel_size,
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
                 max_relative_position=None,
                 lsh_bucket_length=4,
                 block_length=128,
                 filter_width=100,
                 mask_right=False,
                 add_relative_to_values=False,
                 return_attn_weights=False,
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
        self.return_attn_weights = return_attn_weights

        conv_partial = partial(layer_factory.get_1d_layer, kernel_size=kernel_size,
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
                               bias_constraint=self.get_constraint('bias'))
        q_filters = attention_dim
        k_filters = attention_dim
        v_filters = output_dim

        if multiquery_attention:
            k_filters //= num_heads
            v_filters //= num_heads

        q_strides = strides
        kv_strides = strides
        attn_strides = 1
        if attention_type == 'LocalizedAttentionLayer1D' and strides != 1:
            kv_strides = 1
            attn_strides = strides

        self.q_layer = conv_partial(type=self.q_type,
                                    filters=q_filters,
                                    strides=q_strides,
                                    dilation_rate=dilation_rate, name='Conv-Q')
        self.k_layer = conv_partial(type=self.k_type,
                                    filters=k_filters,
                                    strides=kv_strides,
                                    dilation_rate=1, name='Conv-K')
        self.v_layer = conv_partial(type=self.v_type,
                                    filters=v_filters,
                                    strides=kv_strides,
                                    dilation_rate=1, name='Conv-V')

        attention_padding = padding
        self.attention_layer = layer_factory.get_1D_attention_layer(
            type=attention_type,
            strides=attn_strides,
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

    def call(self, inputs, training=None, mask=None):
        x = inputs
        q = self.q_layer(x, training=training, mask=mask)
        k = self.k_layer(x, training=training, mask=mask)
        v = self.v_layer(x, training=training, mask=mask)

        if mask is not None:
            q_mask = mask
            if self.q_layer.supports_masking:
                q_mask = self.q_layer.compute_output_mask(x, mask=mask)
            k_mask = mask
            if self.k_layer.supports_masking:
                k_mask = self.k_layer.compute_output_mask(x, mask=mask)
            v_mask = mask
            if self.v_layer.supports_masking:
                v_mask = self.v_layer.compute_output_mask(x, mask=mask)
            mask = [q_mask, k_mask, v_mask]
        x, weights = self.attention_layer([q, k, v], mask=mask, training=training)

        x = super(SelfAttentionBlock1D, self).call(x, training=training)
        if self.return_attn_weights:
            return x, weights
        return x

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

    def compute_output_shape(self, input_shape):
        qs = self.q_layer.compute_output_shape(input_shape)
        ks = self.k_layer.compute_output_shape(input_shape)
        vs = self.v_layer.compute_output_shape(input_shape)
        output_shape = self.attention_layer.compute_output_shape([qs, ks, vs])
        if self.return_attn_weights:
            return output_shape
        return output_shape[0]

    def compute_output_signature(self, input_signature):
        """Compute the output tensor signature of the layer based on the inputs.

        Unlike a TensorShape object, a TensorSpec object contains both shape
        and dtype information for a tensor. This method allows layers to provide
        output dtype information if it is different from the input dtype.
        For any layer that doesn't implement this function,
        the framework will fall back to use `compute_output_shape`, and will
        assume that the output dtype matches the input dtype.

        Args:
          input_signature: Single TensorSpec or nested structure of TensorSpec
            objects, describing a candidate input for the layer.

        Returns:
          Single TensorSpec or nested structure of TensorSpec objects, describing
            how the layer would transform the provided input.

        Raises:
          TypeError: If input_signature contains a non-TensorSpec object.
        """

        def check_type_return_shape(s):
            if not isinstance(s, tensor_spec.TensorSpec):
                raise TypeError(
                    'Only TensorSpec signature types are supported, '
                    'but saw signature signature entry: {}.'.format(s))
            return s.shape

        input_shape = nest.map_structure(check_type_return_shape, input_signature)
        output_shape = self.compute_output_shape(input_shape)
        dtype = self._compute_dtype
        if dtype is None:
            input_dtypes = [s.dtype for s in nest.flatten(input_signature)]
            # Default behavior when self.dtype is None, is to use the first input's
            # dtype.
            dtype = input_dtypes[0]
        return nest.map_structure(
            lambda s: tensor_spec.TensorSpec(dtype=dtype, shape=s),
            output_shape)

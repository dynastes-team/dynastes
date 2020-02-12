from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.layers as tfkl

from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.util.layer_util import call_masked as cm


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class CausalDilatedWrapper1D(DynastesBaseLayer):

    def __init__(self,
                 d_in,
                 processing_block,
                 kernel_size=3,
                 dilation_rate=1,
                 **kwargs):

        self.d_in = d_in
        self.processing_block = processing_block
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        super(CausalDilatedWrapper1D, self).__init__(**kwargs)

    def request_cache(self, batch_size, **kwargs):

        return {'input_queue': tf.zeros([batch_size, self.kernel_size * self.dilation_rate, self.d_in]),
                'mask_queue': tf.cast(tf.zeros([batch_size, self.kernel_size * self.dilation_rate]), tf.bool)}

    def _call(self, inputs, training=None, cache=None, mask=None, **kwargs):

        if cache is not None:
            inputs = tf.concat([cache['input_queue'], inputs], axis=1)
            if mask is not None:
                mask = tf.concat([cache['mask_queue'], mask], axis=1)

        x, _ = cm(self.processing_block, inputs=inputs, mask=mask, training=training, **kwargs)

        if cache is not None:
            cache['input_queue'] = inputs[:, -(self.kernel_size * self.dilation_rate):, :]
            if mask is not None:
                cache['mask_queue'] = mask[:, -(self.kernel_size * self.dilation_rate):]
                mask = mask[:, -1:]
            return x[:, -1:, :], mask
        return x, mask

    def call(self, inputs, training=None, cache=None, mask=None, **kwargs):
        x, _ = self._call(inputs, training=training, cache=cache, mask=mask, **kwargs)
        return x

    def call_masked(self, inputs, training=None, cache=None, mask=None, **kwargs):
        return self._call(inputs, training=training, cache=cache, mask=mask, **kwargs)

    def compute_output_shape(self, input_shape):
        return self.processing_block.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.processing_block.compute_mask(inputs, mask)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class CausalDilatedBlockStack(tfkl.Layer):
    def __init__(self,
                 blocks,
                 **kwargs):
        super(CausalDilatedBlockStack, self).__init__(**kwargs)
        self.blocks = blocks

    def request_cache(self, batch_size=1):
        cache = {}
        for i, block in enumerate(self.blocks):
            try:
                block_cache = block.request_cache(batch_size=batch_size)
            except:
                block_cache = None
            cache[i] = block_cache
        return cache

    def _call(self, inputs, mask=None, cache=None, training=None, **kwargs):
        if 'decode_loop_step' in kwargs:
            _ = kwargs.pop('decode_loop_step')
        x = inputs
        for i, block in enumerate(self.blocks):
            if cache is not None:
                block_cache = cache[i]
            else:
                block_cache = None
            x, mask = cm(block, x, training=training, mask=mask, cache=block_cache, **kwargs)
        return x

    def call(self, inputs, mask=None, cache=None, training=None, **kwargs):
        x, _ = self._call(inputs, mask=mask, cache=cache, training=training, **kwargs)
        return x

    def call_masked(self, inputs, mask=None, cache=None, training=None, **kwargs):
        return self._call(inputs, mask=mask, cache=cache, training=training, **kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for i, block in enumerate(self.blocks):
            output_shape = block.compute_output_shape(output_shape)
        return output_shape

    def compute_mask(self, inputs, mask=None):
        for i, block in enumerate(self.blocks):
            mask = block.compute_mask(inputs, mask)
        return mask

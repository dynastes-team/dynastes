from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras.layers as tfkl

from dynastes.ops.localized_attention import localized_attention_1d, localized_attention_2d


class LocalizedAttentionLayer1D(tfkl.Layer):

    def __init__(self,
                 kernel_size=3,
                 num_heads=1,
                 strides=1,
                 dilation_rate=1,
                 padding='same',
                 preshaped_q=True, **kwargs):
        """
            Args:
                kernel_size: size of patches to perform localized attention within
                num_heads: number of attention heads
                strides: the strides of the patch window, strides 2 halves output
                dilation_rate: the dilation_rate of the patch window
                padding: one of 'same' or 'valid'
                preshaped_q: True if q matches strided and padded kv
                    ex: kv: [B, 4, C]
                        strides = 2
                        q must be [B,2,C]
        """
        super(LocalizedAttentionLayer1D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.preshaped_q = preshaped_q

    def call(self, q, k, v):
        if type(q) == list:
            if len(q) == 3:
                q, k, v = q
            elif len(q) == 4:
                q, k, v, mask = q
            else:
                raise SyntaxError
        return localized_attention_1d(q=q, k=k, v=v,
                                      num_heads=self.num_heads,
                                      strides=self.strides,
                                      dilation_rate=self.dilation_rate,
                                      padding=self.padding,
                                      preshaped_q=self.preshaped_q)

    def get_config(self):
        config = {'kernel_size': self.kernel_size,
                  'num_heads': self.num_heads,
                  'strides': self.strides,
                  'dilation_rate': self.dilation_rate,
                  'padding': self.padding,
                  'preshaped_q': self.preshaped_q}
        base_config = super(LocalizedAttentionLayer1D, self).get_config()
        return {**base_config, **config}


class LocalizedAttentionLayer2D(tfkl.Layer):

    def __init__(self,
                 kernel_size=(3, 3),
                 num_heads=1,
                 strides=(1, 1),
                 dilation_rate=(1, 1),
                 padding='same',
                 preshaped_q=True, **kwargs):
        """
            Args:
                kernel_size: size of patches to perform localized attention within
                num_heads: number of attention heads
                strides: the strides of the patch window, strides 2 halves output
                dilation_rate: the dilation_rate of the patch window
                padding: one of 'same' or 'valid'
                preshaped_q: True if q matches strided and padded kv
                    ex: kv: [B, 4, 4, C]
                        strides = (2,2)
                        q must be [B,2,2,C]
        """
        super(LocalizedAttentionLayer2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.preshaped_q = preshaped_q

    def call(self, q, k, v):
        if type(q) == list:
            if len(q) == 3:
                q, k, v = q
            elif len(q) == 4:
                q, k, v, mask = q
            else:
                raise SyntaxError
        return localized_attention_2d(q=q, k=k, v=v,
                                      num_heads=self.num_heads,
                                      strides=self.strides,
                                      dilation_rate=self.dilation_rate,
                                      padding=self.padding,
                                      preshaped_q=self.preshaped_q)

    def get_config(self):
        config = {'kernel_size': self.kernel_size,
                  'num_heads': self.num_heads,
                  'strides': self.strides,
                  'dilation_rate': self.dilation_rate,
                  'padding': self.padding,
                  'preshaped_q': self.preshaped_q}
        base_config = super(LocalizedAttentionLayer2D, self).get_config()
        return {**base_config, **config}

del absolute_import
del division
del print_function
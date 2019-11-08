from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras.layers as tfkl

from dynastes.util import localized_attention


class LocalizedAttentionLayer1D(tfkl.Layer):

    def __init__(self,
                 patch_size=3,
                 num_heads=1,
                 stride=1,
                 dilation=1,
                 padding='same',
                 preshaped_q=True, **kwargs):
        """
            Args:
                patch_size: size of patches to perform localized attention within
                num_heads: number of attention heads
                strides: the stride of the patch window, stride 2 halves output
                dilations: the dilation of the patch window
                padding: one of 'same' or 'valid'
                preshaped_q: True if q matches strided and padded kv
                    ex: kv: [B, 4, C]
                        stride = 2
                        q must be [B,2,C]
        """
        super(LocalizedAttentionLayer1D, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.stride = stride
        self.dilation = dilation
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
        return localized_attention.localized_attention_1d(q=q, k=k, v=v,
                                                          num_heads=self.num_heads,
                                                          stride=self.stride,
                                                          dilation=self.dilation,
                                                          padding=self.padding,
                                                          preshaped_q=self.preshaped_q)

    def get_config(self):
        config = {'patch_size': self.patch_size,
                  'num_heads': self.num_heads,
                  'stride': self.stride,
                  'dilation': self.dilation,
                  'padding': self.padding,
                  'preshaped_q': self.preshaped_q}
        base_config = super(LocalizedAttentionLayer1D, self).get_config()
        return {**base_config, **config}


class LocalizedAttentionLayer2D(tfkl.Layer):

    def __init__(self,
                 patch_size=(3, 3),
                 num_heads=1,
                 strides=(1, 1),
                 dilations=(1, 1),
                 padding='same',
                 preshaped_q=True, **kwargs):
        """
            Args:
                patch_size: size of patches to perform localized attention within
                num_heads: number of attention heads
                strides: the stride of the patch window, stride 2 halves output
                dilations: the dilation of the patch window
                padding: one of 'same' or 'valid'
                preshaped_q: True if q matches strided and padded kv
                    ex: kv: [B, 4, 4, C]
                        strides = (2,2)
                        q must be [B,2,2,C]
        """
        super(LocalizedAttentionLayer2D, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.strides = strides
        self.dilations = dilations
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
        return localized_attention.localized_attention_2d(q=q, k=k, v=v,
                                                          num_heads=self.num_heads,
                                                          strides=self.strides,
                                                          dilations=self.dilations,
                                                          padding=self.padding,
                                                          preshaped_q=self.preshaped_q)

    def get_config(self):
        config = {'patch_size': self.patch_size,
                  'num_heads': self.num_heads,
                  'strides': self.strides,
                  'dilations': self.dilations,
                  'padding': self.padding,
                  'preshaped_q': self.preshaped_q}
        base_config = super(LocalizedAttentionLayer2D, self).get_config()
        return {**base_config, **config}

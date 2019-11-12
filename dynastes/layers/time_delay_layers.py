from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import activations

from dynastes.layers import ActivatedKernelBiasBaseLayer
from dynastes.ops.time_delay_ops import time_delay_nn_1d


class _TimeDelayLayer(ActivatedKernelBiasBaseLayer):
    def __init__(self,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(_TimeDelayLayer, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            **kwargs)

    def get_config(self):
        config = {}
        base_config = super(_TimeDelayLayer, self).get_config()
        return {**base_config, **config}


class TimeDelayLayer1D(_TimeDelayLayer):
    def __init__(self,
                 filters,
                 kernel_size=5,
                 strides=1,
                 dilation_rate=1,
                 padding='same',
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(TimeDelayLayer1D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            **kwargs)

        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.filters = filters
        self.padding = padding

    def build(self, input_shape):
        self.input_dim = int(input_shape[-1])
        self.build_kernel([self.input_dim * self.kernel_size, self.filters])
        self.build_bias(self.filters)
        super(TimeDelayLayer1D, self).build(input_shape)

    def call(self, x, training=None):
        x = time_delay_nn_1d(x, self.get_weight('kernel', training=training),
                             kernel_size=self.kernel_size,
                             strides=self.strides,
                             dilation_rate=self.dilation_rate)
        return super(TimeDelayLayer1D, self).call(x, training=training)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
        }
        base_config = super(TimeDelayLayer1D, self).get_config()
        return {**base_config, **config}


class _MultiTimeDelayLayer(ActivatedKernelBiasBaseLayer):

    def __init__(self,
                 activation=None,
                 use_bias=True,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(_MultiTimeDelayLayer, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            trainable=trainable,
            name=name,
            **kwargs)

    def build_child_layer(self, filters):
        return TimeDelayLayer1D(filters,
                                kernel_size=self.kernel_size,
                                strides=self.strides,
                                dilation_rate=self.dilation_rate,
                                padding=self.padding,
                                use_bias=False,
                                kernel_initializer=self.get_initializer('kernel'),
                                kernel_regularizer=self.get_regularizer('kernel'),
                                kernel_constraint=self.get_constraint('kernel'))

    def get_config(self):
        config = {}
        base_config = super(_MultiTimeDelayLayer, self).get_config()
        return {**base_config, **config}


class DepthGroupwiseTimeDelayLayer1D(_MultiTimeDelayLayer):
    def __init__(self,
                 depth_multiplier=1,
                 kernel_size=5,
                 strides=1,
                 group_size=1,
                 dilation_rate=1,
                 grouped=False,
                 padding='same',
                 activation=None,
                 use_bias=True,
                 input_dim=None,
                 **kwargs):
        super(DepthGroupwiseTimeDelayLayer1D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            **kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.group_size = group_size
        self.grouped = grouped
        self.padding = padding
        self.input_dim = input_dim
        if input_dim is not None:
            self.init_layers()

    def init_layers(self):
        self.n_groups = self.input_dim // self.group_size
        assert ((self.input_dim % self.group_size) == 0)
        if self.grouped:
            self.layer = super(DepthGroupwiseTimeDelayLayer1D, self).build_child_layer(
                self.depth_multiplier * self.group_size)
        else:
            self.layers = [super(DepthGroupwiseTimeDelayLayer1D, self).build_child_layer(self.depth_multiplier) for _ in
                           range(self.n_groups)]

    def build(self, input_shape):
        if self.input_dim is None:
            self.input_dim = int(input_shape[-1])
            self.init_layers()

        super(DepthGroupwiseTimeDelayLayer1D, self).build_bias(self.input_dim * self.depth_multiplier)
        super(DepthGroupwiseTimeDelayLayer1D, self).build(input_shape)

    def call(self, x, training=None):
        splits = tf.split(x, self.n_groups, -1)
        if self.grouped:
            outs = [self.layer(xs) for xs in splits]
        else:
            outs = [self.layers[i](xs) for i, xs in enumerate(splits)]
        return super(DepthGroupwiseTimeDelayLayer1D, self).call(tf.concat(outs, axis=-1), training=training)

    def get_config(self):
        config = {
            'depth_multiplier': self.depth_multiplier,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'group_size': self.group_size,
            'grouped': self.grouped,
            'padding': self.padding,
            'input_dim': self.input_dim,
        }
        base_config = super(DepthGroupwiseTimeDelayLayer1D, self).get_config()
        return {**base_config, **config}


class TimeDelayLayerFake2D(_MultiTimeDelayLayer):
    def __init__(self,
                 filters,
                 kernel_size=5,
                 strides=1,
                 dilation_rate=1,
                 padding='same',
                 activation=None,
                 use_bias=True,
                 input_dim=None,
                 **kwargs):
        super(TimeDelayLayerFake2D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.input_dim = input_dim
        self.layers = None
        if self.input_dim is not None:
            self.init_layers()

    def init_layers(self):
        self.layers = [super(TimeDelayLayerFake2D, self).build_child_layer(self.filters) for _ in
                       range(self.input_dim)]

    def build(self, input_shape):
        if self.input_dim is None:
            self.input_dim = int(input_shape[-2])
            self.init_layers()

        super(TimeDelayLayerFake2D, self).build_bias(self.filters)
        super(TimeDelayLayerFake2D, self).build(input_shape)

    def call(self, x, **kwargs):
        splits = tf.split(x, self.input_dim, -2)
        outs = [tf.expand_dims(self.layers[i](tf.squeeze(xs, -2)), axis=-2) for i, xs in enumerate(splits)]
        return super(TimeDelayLayerFake2D, self).call(tf.concat(outs, axis=-2), **kwargs)

    def get_config(self):
        config = {'filters': self.filters, 'kernel_size': self.kernel_size, 'strides': self.strides,
                  'dilation_rate': self.dilation_rate, 'padding': self.padding, 'input_dim': self.input_dim}
        base_config = super(TimeDelayLayerFake2D, self).get_config()
        return {**base_config, **config}


class DepthGroupwiseTimeDelayLayerFake2D(_MultiTimeDelayLayer):
    def __init__(self,
                 depth_multiplier=1,
                 kernel_size=5,
                 strides=1,
                 dilation_rate=1,
                 grouped=False,
                 group_size=1,
                 padding='same',
                 activation=None,
                 use_bias=True,
                 input_dim=None,
                 **kwargs):

        super(DepthGroupwiseTimeDelayLayerFake2D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.layers = None
        self.input_dim = input_dim
        self.grouped = grouped
        self.group_size = group_size
        if self.input_dim is not None:
            self.init_layers()

    def build_child_layer(self, depth_multiplier):
        return DepthGroupwiseTimeDelayLayer1D(depth_multiplier,
                                              grouped=self.grouped,
                                              group_size=self.group_size,
                                              kernel_size=self.kernel_size,
                                              strides=self.strides,
                                              dilation_rate=self.dilation_rate,
                                              padding=self.padding,
                                              use_bias=False,
                                              kernel_initializer=self.get_initializer('kernel'),
                                              kernel_regularizer=self.get_regularizer('kernel'),
                                              kernel_constraint=self.get_constraint('kernel'))

    def init_layers(self):
        self.layers = [self.build_child_layer(self.depth_multiplier) for _ in
                       range(self.input_dim)]

    def build(self, input_shape):
        if self.layers is None:
            self.input_dim = int(input_shape[-2])
            self.init_layers()
        self.input_channels = int(input_shape[-1])
        super(DepthGroupwiseTimeDelayLayerFake2D, self).build_bias(self.input_channels)
        super(DepthGroupwiseTimeDelayLayerFake2D, self).build(input_shape)

    def call(self, x, **kwargs):
        splits = tf.split(x, self.input_dim, -2)
        outs = [tf.expand_dims(self.layers[i](tf.squeeze(xs, -2)), axis=-2) for i, xs in enumerate(splits)]
        return super(DepthGroupwiseTimeDelayLayerFake2D, self).call(tf.concat(outs, axis=-2), **kwargs)

    def get_config(self):
        config = {'depth_multiplier': self.depth_multiplier,
                  'kernel_size': self.kernel_size,
                  'grouped': self.grouped,
                  'group_size': self.group_size,
                  'strides': self.strides,
                  'dilation_rate': self.dilation_rate,
                  'padding': self.padding,
                  'input_dim': self.input_dim}
        base_config = super(DepthGroupwiseTimeDelayLayerFake2D, self).get_config()
        return {**base_config, **config}


del absolute_import
del division
del print_function

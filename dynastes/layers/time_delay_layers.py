from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import nn


class _TimeDelayLayer(tfkl.Layer):
    def __init__(self,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(_TimeDelayLayer, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build_kernel(self, shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

    def build_bias(self, output_dim):
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

    def call(self, x):
        if self.use_bias:
            x = nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            return self.activation(x)
        return x

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_TimeDelayLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeDelayLayer1D(_TimeDelayLayer):
    def __init__(self,
                 output_dim,
                 context_size=5,
                 stride=1,
                 dilation=1,
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
        super(TimeDelayLayer1D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

        self.context_size = context_size
        self.stride = stride
        self.dilation = dilation
        self.output_dim = output_dim
        self.padding = padding

    def build(self, input_shape):
        self.input_dim = int(input_shape[-1])
        self.build_kernel([self.input_dim * self.context_size, self.output_dim])
        self.build_bias(self.output_dim)
        super(TimeDelayLayer1D, self).build(input_shape)

    def call(self, x):
        x = tf.expand_dims(x, -1)
        x = tf.image.extract_patches(x,
                                     sizes=[1, self.context_size, self.input_dim, 1],
                                     strides=[1, self.stride, self.input_dim, 1],
                                     rates=[1, self.dilation, 1, 1],
                                     padding=self.padding.upper())
        x = tf.squeeze(x, -2)
        x = tf.matmul(x, self.kernel)
        return super(TimeDelayLayer1D, self).call(x)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'context_size': self.context_size,
            'stride': self.stride,
            'padding': self.padding,
        }
        base_config = super(TimeDelayLayer1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class _MultiTimeDelayLayer(tfkl.Layer):

    def __init__(self,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(_MultiTimeDelayLayer, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

    def build_child_layer(self, filters):
        return TimeDelayLayer1D(filters,
                                context_size=self.context_size,
                                stride=self.stride,
                                dilation=self.dilation,
                                padding=self.padding,
                                use_bias=False,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer,
                                kernel_constraint=self.kernel_constraint)

    def build_bias(self, output_dim):
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

    def call(self, x):
        if self.use_bias:
            x = nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            return self.activation(x)
        return x

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_MultiTimeDelayLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DepthGroupwiseTimeDelayLayer1D(_MultiTimeDelayLayer):
    def __init__(self,
                 output_mul=1,
                 context_size=5,
                 stride=1,
                 group_size=1,
                 dilation=1,
                 grouped=False,
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
                 input_dim=None,
                 **kwargs):
        super(DepthGroupwiseTimeDelayLayer1D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
        self.context_size = context_size
        self.stride = stride
        self.dilation = dilation
        self.output_mul = output_mul
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
                self.output_mul * self.group_size)
        else:
            self.layers = [super(DepthGroupwiseTimeDelayLayer1D, self).build_child_layer(self.output_mul) for _ in
                           range(self.n_groups)]

    def build(self, input_shape):
        if self.input_dim is None:
            self.input_dim = int(input_shape[-1])
            self.init_layers()

        super(DepthGroupwiseTimeDelayLayer1D, self).build_bias(self.input_dim * self.output_mul)
        super(DepthGroupwiseTimeDelayLayer1D, self).build(input_shape)

    def call(self, x):
        splits = tf.split(x, self.n_groups, -1)
        if self.grouped:
            outs = [self.layer(xs) for xs in splits]
        else:
            outs = [self.layers[i](xs) for i, xs in enumerate(splits)]
        return super(DepthGroupwiseTimeDelayLayer1D, self).call(tf.concat(outs, axis=-1))

    def get_config(self):
        config = {
            'output_mul': self.output_mul,
            'context_size': self.context_size,
            'stride': self.stride,
            'group_size': self.group_size,
            'grouped': self.grouped,
            'padding': self.padding,
            'input_dim': self.input_dim,
        }
        base_config = super(DepthGroupwiseTimeDelayLayer1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeDelayLayerFake2D(_MultiTimeDelayLayer):
    def __init__(self,
                 output_dim,
                 context_size=5,
                 stride=1,
                 dilation=1,
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
                 input_dim=None,
                 **kwargs):
        super(TimeDelayLayerFake2D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=kernel_constraint,
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
        self.output_dim = output_dim
        self.context_size = context_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding.upper()
        self.input_dim = input_dim
        self.layers = None
        if self.input_dim is not None:
            self.init_layers()

    def init_layers(self):
        self.layers = [super(TimeDelayLayerFake2D, self).build_child_layer(self.output_dim) for _ in
                       range(self.input_dim)]

    def build(self, input_shape):
        if self.input_dim is None:
            self.input_dim = int(input_shape[-2])
            self.init_layers()

        super(TimeDelayLayerFake2D, self).build_bias(self.output_dim)
        super(TimeDelayLayerFake2D, self).build(input_shape)

    def call(self, x):
        splits = tf.split(x, self.input_dim, -2)
        outs = [tf.expand_dims(self.layers[i](tf.squeeze(xs, -2)), axis=-2) for i, xs in enumerate(splits)]
        return super(TimeDelayLayerFake2D, self).call(tf.concat(outs, axis=-2))

    def get_config(self):
        config = {'output_dim': self.output_dim, 'context_size': self.context_size, 'stride': self.stride,
                  'dilation': self.dilation, 'padding': self.padding, 'input_dim': self.input_dim}
        base_config = super(TimeDelayLayerFake2D, self).get_config()
        return {**base_config, **config}


class DepthGroupwiseTimeDelayLayerFake2D(_MultiTimeDelayLayer):
    def __init__(self,
                 output_mul=1,
                 context_size=5,
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
                 input_dim=None,
                 **kwargs):

        super(DepthGroupwiseTimeDelayLayerFake2D, self).__init__(
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=kernel_constraint,
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
        self.output_mul = output_mul
        self.context_size = context_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.layers = None
        self.input_dim = input_dim
        self.grouped = grouped
        self.group_size = group_size
        if self.input_dim is not None:
            self.init_layers()

    def build_child_layer(self, output_mul):
        return DepthGroupwiseTimeDelayLayer1D(output_mul,
                                              grouped=self.grouped,
                                              group_size=self.group_size,
                                              context_size=self.context_size,
                                              stride=self.stride,
                                              dilation=self.dilation,
                                              padding=self.padding,
                                              use_bias=False,
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=self.kernel_regularizer,
                                              kernel_constraint=self.kernel_constraint)

    def init_layers(self):
        self.layers = [self.build_child_layer(self.output_mul) for _ in
                       range(self.input_dim)]

    def build(self, input_shape):
        if self.layers is None:
            self.input_dim = int(input_shape[-2])
            self.init_layers()
        self.input_channels = int(input_shape[-1])
        super(DepthGroupwiseTimeDelayLayerFake2D, self).build_bias(self.input_channels)
        super(DepthGroupwiseTimeDelayLayerFake2D, self).build(input_shape)

    def call(self, x):
        splits = tf.split(x, self.input_dim, -2)
        outs = [tf.expand_dims(self.layers[i](tf.squeeze(xs, -2)), axis=-2) for i, xs in enumerate(splits)]
        return super(DepthGroupwiseTimeDelayLayerFake2D, self).call(tf.concat(outs, axis=-2))

    def get_config(self):
        config = {'output_mul': self.output_mul,
                  'context_size': self.context_size,
                  'grouped': self.grouped,
                  'group_size': self.group_size,
                  'stride': self.stride,
                  'dilation': self.dilation,
                  'padding': self.padding,
                  'input_dim': self.input_dim}
        base_config = super(DepthGroupwiseTimeDelayLayerFake2D, self).get_config()
        return {**base_config, **config}

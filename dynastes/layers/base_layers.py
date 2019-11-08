import tensorflow.keras.layers as tfkl

from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import nn


class DynastesBaseLayer(tfkl.Layer):
    def __init__(self,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(DynastesBaseLayer, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.weights_dict = {}

    def add_weight(self,
                 name=None,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 constraint=None,
                 partitioner=None,
                 use_resource=None,
                 **kwargs):
        weight = super(DynastesBaseLayer, self).add_weight(name=name,
                                                           shape=shape,
                                                           dtype=dtype,
                                                           initializer=initializer,
                                                           regularizer=regularizer,
                                                           trainable=trainable,
                                                           constraint=constraint,
                                                           partitioner=partitioner,
                                                           use_resource=use_resource,
                                                           **kwargs)
        self.weights_dict[name] = weight
        return weight

    def get_weight(self, name):
        return self.weights_dict[name]

    def get_config(self):
        config = {
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
        }
        base_config = super(DynastesBaseLayer, self).get_config()
        return {**base_config, **config}


class ActivatedKernelBiasBaseLayer(DynastesBaseLayer):
    def __init__(self,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='he_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ActivatedKernelBiasBaseLayer, self).__init__(
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
        self.add_weight(
            name='kernel',
            shape=shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

    def build_bias(self, output_dim):
        if self.use_bias:
            self.add_weight(
                name='bias',
                shape=(output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

    def call(self, x, **kwargs):
        if self.use_bias:
            x = nn.bias_add(x, self.get_weight('bias'), data_format='NHWC')
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
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ActivatedKernelBiasBaseLayer, self).get_config()
        return {**base_config, **config}

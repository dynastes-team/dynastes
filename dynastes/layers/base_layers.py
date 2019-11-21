import copy

import tensorflow.keras.layers as tfkl
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.ops import nn

from dynastes import normalizers
from dynastes import regularizers


def _get_regularizers_from_keywords(kwargs):
    _initializers = {}
    _regularizers = {}
    _constraints = {}
    _normalizers = {}

    kwarg_keys = copy.copy(list(kwargs.keys()))

    for kwarg in kwarg_keys:
        if kwarg.endswith('initializer'):
            _initializers[kwarg.split('_initializer')[0]] = initializers.get(kwargs.pop(kwarg, None))
        elif kwarg.endswith('regularizer'):
            if kwarg != 'activity_regularizer':
                _regularizers[kwarg.split('_regularizer')[0]] = regularizers.get(kwargs.pop(kwarg))
        elif kwarg.endswith('constraint'):
            _constraints[kwarg.split('_constraint')[0]] = constraints.get(kwargs.pop(kwarg))
        elif kwarg.endswith('normalizer'):
            _normalizers[kwarg.split('_normalizer')[0]] = normalizers.get(kwargs.pop(kwarg))

    return _initializers, _regularizers, _constraints, _normalizers


class DynastesBaseLayer(tfkl.Layer):
    def __init__(self,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        self.weights_dict = {}
        self.initializers, self.regularizers, self.constraints, self.normalizers = _get_regularizers_from_keywords(kwargs)

        super(DynastesBaseLayer, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

    def get_initializer(self, name):
        if name not in self.initializers:
            if name == 'kernel':
                self.initializers['kernel'] = initializers.get('he_uniform')
            elif name == 'bias':
                self.initializers['bias'] = initializers.get('zeros')
            else:
                self.initializers[name] = initializers.get('glorot_uniform')

        return self.initializers[name]

    def get_regularizer(self, name):
        if name not in self.regularizers:
            self.regularizers[name] = regularizers.get(None)
        return self.regularizers[name]

    def get_constraint(self, name):
        if name not in self.constraints:
            self.constraints[name] = constraints.get(None)
        return self.constraints[name]

    def add_weight(self,
                   name=None,
                   shape=None,
                   trainable=None,
                   partitioner=None,
                   initializer=None,
                   regularizer=None,
                   constraint=None,
                   dtype=None,
                   use_resource=None,
                   **kwargs):
        if initializer is not None:
            self.initializers[name] = initializers.get(initializer)
        if regularizer is not None:
            self.regularizers[name] = regularizers.get(regularizer)
        if constraint is not None:
            self.constraints[name] = constraints.get(constraint)
        weight = super(DynastesBaseLayer, self).add_weight(name=name,
                                                           shape=shape,
                                                           initializer=self.get_initializer(name),
                                                           regularizer=self.get_regularizer(name),
                                                           trainable=trainable,
                                                           constraint=self.get_constraint(name),
                                                           partitioner=partitioner,
                                                           use_resource=use_resource,
                                                           **kwargs)
        self.weights_dict[name] = weight
        return weight

    def get_weight(self, name, training=None):
        w = self.weights_dict[name]
        if name in self.normalizers:
            w = self.normalizers[name](w, training=training)
        return w

    def get_config(self):
        config = {
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
        }
        for name, initializer in self.initializers.items():
            config[name + '_initializer'] = initializers.serialize(initializer)
        for name, regularizer in self.regularizers.items():
            config[name + '_regularizer'] = regularizers.serialize(regularizer)
        for name, constraint in self.constraints.items():
            config[name + '_constraint'] = constraints.serialize(constraint)
        for name, normalizer in self.normalizers.items():
            config[name + '_normalizer'] = normalizers.serialize(normalizer)

        base_config = super(DynastesBaseLayer, self).get_config()
        return {**base_config, **config}


class ActivatedKernelBiasBaseLayer(DynastesBaseLayer):
    def __init__(self,
                 activation=None,
                 use_bias=True,
                 activity_regularizer=None,
                 **kwargs):
        super(ActivatedKernelBiasBaseLayer, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build_kernel(self, shape):
        self.add_weight(
            name='kernel',
            shape=shape,
            trainable=True,
            dtype=self.dtype)

    def build_bias(self, output_dim):
        if self.use_bias:
            self.add_weight(
                name='bias',
                shape=(output_dim,),
                trainable=True,
                dtype=self.dtype)

    def call(self, x, training=None):
        if self.use_bias:
            x = nn.bias_add(x, self.get_weight('bias', training=training), data_format='NHWC')
        if self.activation is not None:
            return self.activation(x)
        return x

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
        }
        base_config = super(ActivatedKernelBiasBaseLayer, self).get_config()
        return {**base_config, **config}

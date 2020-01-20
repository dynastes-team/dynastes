import copy

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops

from dynastes import activations
from dynastes import regularizers
from dynastes import weight_normalizers


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class _WscaleInitializer(tfk.initializers.Initializer):

    def __init__(self,
                 initializer,
                 lrmul=1.,
                 **kwargs):
        super(_WscaleInitializer, self).__init__(**kwargs)
        self.initializer = tfk.initializers.get(initializer)
        self.lrmul = lrmul

    def __call__(self, shape, dtype=None):
        return self.initializer(shape, dtype=dtype) / self.lrmul

    def get_config(self):
        config = {
            'initializer': tfk.initializers.serialize(self.initializer),
            'lrmul': self.lrmul,
        }
        base_config = super(_WscaleInitializer, self).get_config()
        return {**base_config, **config}


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
                _regularizers[kwarg.split('_regularizer')[0]] = regularizers.get(kwargs.pop(kwarg, None))
        elif kwarg.endswith('constraint'):
            _constraints[kwarg.split('_constraint')[0]] = constraints.get(kwargs.pop(kwarg, None))
        elif kwarg.endswith('normalizer'):
            _normalizers[kwarg.split('_normalizer')[0]] = weight_normalizers.get(kwargs.pop(kwarg, None))

    return _initializers, _regularizers, _constraints, _normalizers


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DynastesBaseLayer(tfkl.Layer):
    def __init__(self,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 use_wscale=False,
                 wlrmul=1.,
                 wgain=np.sqrt(2),
                 supports_caching=False,
                 mask_threshold=0.5,
                 **kwargs):
        self.weights_dict = {}
        self.initializers, self.regularizers, self.constraints, self.normalizers = _get_regularizers_from_keywords(
            kwargs)
        self.use_wscale = use_wscale
        self.lrmul = wlrmul
        self.gain = wgain
        self.supports_caching = supports_caching
        self.mask_threshold = mask_threshold
        super(DynastesBaseLayer, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.supports_masking = True

    def request_cache(self, batch_size, **kwargs):
        pass

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
        _initializer = self.get_initializer(name)
        if self.use_wscale:
            _initializer = _WscaleInitializer(_initializer, lrmul=self.lrmul)
            self.initializers[name] = _initializer
            if name in self.normalizers and self.normalizers[name] is not None:
                self.normalizers[name] = weight_normalizers.WscaleNormalizer(next_layer=self.normalizers[name],
                                                                             lrmul=self.lrmul,
                                                                             gain=self.gain)
            else:
                self.normalizers[name] = weight_normalizers.WscaleNormalizer(lrmul=self.lrmul, gain=self.gain)

        weight = super(DynastesBaseLayer, self).add_weight(name=name,
                                                           shape=shape,
                                                           initializer=_initializer,
                                                           regularizer=self.get_regularizer(name),
                                                           trainable=trainable,
                                                           constraint=self.get_constraint(name),
                                                           partitioner=partitioner,
                                                           use_resource=use_resource,
                                                           **kwargs)
        if name in self.normalizers:
            if self.normalizers[name] is not None:
                self.normalizers[name].build(shape)
        self.weights_dict[name] = weight
        return weight

    def get_weight(self, name, training=None):
        w = self.weights_dict[name]
        if name in self.normalizers and self.normalizers[name] is not None:
            w = self.normalizers[name](w, training=training)
        return w

    def get_config(self):
        config = {
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'use_wscale': self.use_wscale,
            'wlrmul': self.lrmul,
            'wgain': self.gain,
            'supports_caching': self.supports_caching,
            'mask_threshold': self.mask_threshold,
        }
        for name, initializer in self.initializers.items():
            config[name + '_initializer'] = initializers.serialize(initializer)
        for name, regularizer in self.regularizers.items():
            config[name + '_regularizer'] = regularizers.serialize(regularizer)
        for name, constraint in self.constraints.items():
            config[name + '_constraint'] = constraints.serialize(constraint)
        for name, normalizer in self.normalizers.items():
            config[name + '_normalizer'] = weight_normalizers.serialize(normalizer)

        base_config = super(DynastesBaseLayer, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
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
        return self.add_weight(
            name='kernel',
            shape=shape,
            trainable=True,
            dtype=self.dtype)

    def build_bias(self, output_dim):
        if self.use_bias:
            return self.add_weight(
                name='bias',
                shape=(output_dim,),
                trainable=True,
                dtype=self.dtype)
        return None

    def post_process_call(self, x, training=None):
        if self.use_bias:
            x = nn.bias_add(x, self.get_weight('bias', training=training), data_format='NHWC')
        if self.activation is not None:
            return self.activation(x)
        return x

    def call(self, x, training=None):
        return self.post_process_call(x, training=training)

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
        }
        base_config = super(ActivatedKernelBiasBaseLayer, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DynastesDense(ActivatedKernelBiasBaseLayer):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: If the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    Example:
    ```python
    # as first layer in a sequential model:
    model = Sequential()
    model.add(Dense(32, input_shape=(16,)))
    # now the model will take as input arrays of shape (*, 16)
    # and output arrays of shape (*, 32)
    # after the first layer, you don't need to specify
    # the size of the input anymore:
    model.add(Dense(32))
    ```
    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 units,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(DynastesDense, self).__init__(**kwargs)
        self.units = int(units)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or tfk.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        self.kernel = self.build_kernel(
            shape=[last_dim, self.units])
        self.build_bias(self.units)
        self.built = True

    def call(self, inputs, training=None):
        rank = len(inputs.shape)
        kernel = self.get_weight('kernel', training=training)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            if tfk.backend.is_sparse(inputs):
                outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, kernel)
            else:
                outputs = gen_math_ops.mat_mul(inputs, kernel)
        return super(DynastesDense, self).call(outputs, training=training)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
        }
        base_config = super(DynastesDense, self).get_config()
        return {**base_config, **config}

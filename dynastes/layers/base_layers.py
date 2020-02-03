import copy

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
import tensorflow.keras.layers as tfkl
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops

from dynastes import activations
from dynastes import regularizers
from dynastes import weight_normalizers
from dynastes.ops import embedding_ops


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


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DynastesEmbedding(DynastesBaseLayer):
    """Turns positive integers (indexes) into dense vectors of fixed size.
    e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`
    This layer can only be used as the first layer in a model.
    Example:
    ```python
    model = Sequential()
    model.add(Embedding(1000, 64, input_length=10))
    # the model will take as input an integer matrix of size (batch,
    # input_length).
    # the largest integer (i.e. word index) in the input should be no larger
    # than 999 (vocabulary size).
    # now model.output_shape == (None, 10, 64), where None is the batch
    # dimension.
    input_array = np.random.randint(1000, size=(32, 10))
    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)
    assert output_array.shape == (32, 10, 64)
    ```
    Arguments:
      input_dim: int > 0. Size of the vocabulary,
        i.e. maximum integer index + 1.
      output_dim: int >= 0. Dimension of the dense embedding.
      embeddings_initializer: Initializer for the `embeddings` matrix.
      embeddings_regularizer: Regularizer function applied to
        the `embeddings` matrix.
      embeddings_constraint: Constraint function applied to
        the `embeddings` matrix.
      mask_zero: Whether or not the input value 0 is a special "padding"
        value that should be masked out.
        This is useful when using recurrent layers
        which may take variable length input.
        If this is `True` then all subsequent layers
        in the model need to support masking or an exception will be raised.
        If mask_zero is set to True, as a consequence, index 0 cannot be
        used in the vocabulary (input_dim should equal size of
        vocabulary + 1).
      input_length: Length of input sequences, when it is constant.
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed).
    Input shape:
      2D tensor with shape: `(batch_size, input_length)`.
    Output shape:
      3D tensor with shape: `(batch_size, input_length, output_dim)`.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 mask_zero=False,
                 input_length=None,
                 symbol_dropout_rate=0.,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        dtype = kwargs.pop('dtype', K.floatx())
        # We set autocast to False, as we do not want to cast floating- point inputs
        # to self.dtype. In call(), we cast to int32, and casting to self.dtype
        # before casting to int32 might cause the int32 values to be different due
        # to a loss of precision.
        kwargs['autocast'] = False

        # Use transposed spectral norm for embedding because... ?
        embedding_norm = kwargs.pop('embedding_normalizer', None)
        if embedding_norm is not None and embedding_norm == 'spectral':
            embedding_norm = 'spectral_t'
        kwargs['embedding_normalizer'] = embedding_norm

        super(DynastesEmbedding, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero
        self.symbol_dropout_rate = symbol_dropout_rate
        self.supports_masking = mask_zero
        self.input_length = input_length
        self._supports_ragged_inputs = True

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Note: most sparse optimizers do not have GPU kernels defined. When
        # building graphs, the placement algorithm is able to place variables on CPU
        # since it knows all kernels using the variable only exist on CPU.
        # When eager execution is enabled, the placement decision has to be made
        # right now. Checking for the presence of GPUs to avoid complicating the
        # TPU codepaths which can handle sparse optimizers.
        # if context.executing_eagerly() and context.context().num_gpus():
        #    with ops.device('cpu:0'):
        self.embedding = self.add_weight(
            name='embedding',
            shape=(self.input_dim, self.output_dim))
        # else:
        #    self.embeddings = self.add_weight(
        #        shape=(self.input_dim, self.output_dim),
        #        initializer=self.embeddings_initializer,
        #        name='embeddings',
        #        regularizer=self.embeddings_regularizer,
        #        constraint=self.embeddings_constraint)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None

        return math_ops.not_equal(inputs, 0)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.input_length is None:
            return input_shape + (self.output_dim,)
        else:
            # input_length can be tuple if input is 3D or higher
            if isinstance(self.input_length, (list, tuple)):
                in_lens = list(self.input_length)
            else:
                in_lens = [self.input_length]
            if len(in_lens) != len(input_shape) - 1:
                raise ValueError('"input_length" is %s, '
                                 'but received input has shape %s' % (str(
                    self.input_length), str(input_shape)))
            else:
                for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
                    if s1 is not None and s2 is not None and s1 != s2:
                        raise ValueError('"input_length" is %s, '
                                         'but received input has shape %s' % (str(
                            self.input_length), str(input_shape)))
                    elif s1 is None:
                        in_lens[i] = s2
            return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)

    def call(self, inputs, training=None):
        dtype = K.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')
        out = embedding_ops.embedding_lookup(inputs, self.embedding,
                                             symbol_dropout_rate=self.symbol_dropout_rate,
                                             dtype=self.dtype)
        return out

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'mask_zero': self.mask_zero,
            'input_length': self.input_length,
            'symbol_dropout_rate': self.symbol_dropout_rate,
        }
        base_config = super(DynastesEmbedding, self).get_config()
        return {**base_config, **config}

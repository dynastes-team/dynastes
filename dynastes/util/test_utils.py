from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.util import tf_inspect

_thread_local_data = threading.local()
_thread_local_data.model_type = None
_thread_local_data.run_eagerly = None
_thread_local_data.experimental_run_tf_function = None
_thread_local_data.saved_model_format = None


def should_run_eagerly():
    """Returns whether the models we are testing should be run eagerly."""
    if _thread_local_data.run_eagerly is None:
        raise ValueError('Cannot call `should_run_eagerly()` outside of a '
                         '`run_eagerly_scope()` or `run_all_keras_modes` '
                         'decorator.')

    return _thread_local_data.run_eagerly and context.executing_eagerly()


@test_util.use_deterministic_cudnn
def layer_test(layer_cls, kwargs=None, input_shape=None, input_dtype=None,
               input_data=None, expected_output=None,
               expected_output_dtype=None, expected_output_shape=None,
               validate_training=True, adapt_data=None):
    """Test routine for a layer with a single input and single output.

    Arguments:
      layer_cls: Layer class object.
      kwargs: Optional dictionary of keyword arguments for instantiating the
        layer.
      input_shape: Input shape tuple.
      input_dtype: Data type of the input data.
      input_data: Numpy array of input data.
      expected_output: Numpy array of the expected output.
      expected_output_dtype: Data type expected for the output.
      expected_output_shape: Shape tuple for the expected shape of the output.
      validate_training: Whether to attempt to validate training on this layer.
        This might be set to False for non-differentiable layers that output
        string or integer values.
      adapt_data: Optional data for an 'adapt' call. If None, adapt() will not
        be tested for this layer. This is only relevant for PreprocessingLayers.

    Returns:
      The output data (Numpy array) returned by the layer, for additional
      checks to be done by the calling code.

    Raises:
      ValueError: if `input_shape is None`.
    """
    if input_data is None:
        if input_shape is None:
            raise ValueError('input_shape is None')
        if not input_dtype:
            input_dtype = 'float32'
        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        input_data = 10 * np.random.random(input_data_shape)
        if input_dtype in [tf.complex64, tf.complex128]:
            input_data = tf.complex(input_data, tf.convert_to_tensor(0.0, dtype=input_data.dtype))
            input_data = tf.cast(input_data, input_dtype).numpy()
        else:
            if input_dtype == 'float':
                input_data -= 0.5
            input_data = input_data.astype(input_dtype)
    elif input_shape is None:
        input_shape = input_data.shape
    if input_dtype is None:
        input_dtype = input_data.dtype
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    # instantiation
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if 'weights' in tf_inspect.getargspec(layer_cls.__init__):
        kwargs['weights'] = weights
        layer = layer_cls(**kwargs)

    # test in functional API
    x = keras.layers.Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)
    if keras.backend.dtype(y) != expected_output_dtype:
        raise AssertionError('When testing layer %s, for input %s, found output '
                             'dtype=%s but expected to find %s.\nFull kwargs: %s' %
                             (layer_cls.__name__,
                              x,
                              keras.backend.dtype(y),
                              expected_output_dtype,
                              kwargs))

    def assert_shapes_equal(expected, actual):
        """Asserts that the output shape from the layer matches the actual shape."""
        if len(expected) != len(actual):
            raise AssertionError(
                'When testing layer %s, for input %s, found output_shape='
                '%s but expected to find %s.\nFull kwargs: %s' %
                (layer_cls.__name__, x, actual, expected, kwargs))

        for expected_dim, actual_dim in zip(expected, actual):
            if isinstance(expected_dim, tensor_shape.Dimension):
                expected_dim = expected_dim.value
            if isinstance(actual_dim, tensor_shape.Dimension):
                actual_dim = actual_dim.value
            if expected_dim is not None and expected_dim != actual_dim:
                raise AssertionError(
                    'When testing layer %s, for input %s, found output_shape='
                    '%s but expected to find %s.\nFull kwargs: %s' %
                    (layer_cls.__name__, x, actual, expected, kwargs))

    if expected_output_shape is not None:
        assert_shapes_equal(tensor_shape.TensorShape(expected_output_shape),
                            y.shape)

    # check shape inference
    model = keras.models.Model(x, y)
    _comp_output_shape = layer.compute_output_shape(
        tensor_shape.TensorShape(input_shape))
    computed_output_shape = tuple(_comp_output_shape.as_list())
    computed_output_signature = layer.compute_output_signature(
        tensor_spec.TensorSpec(shape=input_shape, dtype=input_dtype))
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    assert_shapes_equal(computed_output_shape, actual_output_shape)
    assert_shapes_equal(computed_output_signature.shape, actual_output_shape)
    if computed_output_signature.dtype != actual_output.dtype:
        raise AssertionError(
            'When testing layer %s, for input %s, found output_dtype='
            '%s but expected to find %s.\nFull kwargs: %s' %
            (layer_cls.__name__, x, actual_output.dtype,
             computed_output_signature.dtype, kwargs))
    if expected_output is not None:
        np.testing.assert_allclose(actual_output, expected_output,
                                   rtol=1e-3, atol=1e-6)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = keras.models.Model.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        np.testing.assert_allclose(output, actual_output, rtol=1e-3, atol=1e-6)

    # test training mode (e.g. useful for dropout tests)
    # Rebuild the model to avoid the graph being reused between predict() and
    # See b/120160788 for more details. This should be mitigated after 2.0.
    if validate_training:
        model = keras.models.Model(x, layer(x))
        if _thread_local_data.run_eagerly is not None:
            model.compile(
                'rmsprop',
                'mse',
                weighted_metrics=['acc'],
                run_eagerly=should_run_eagerly())
        else:
            model.compile('rmsprop', 'mse', weighted_metrics=['acc'])
        model.train_on_batch(input_data, actual_output)

    # test as first layer in Sequential API
    layer_config = layer.get_config()
    layer_config['batch_input_shape'] = input_shape
    layer = layer.__class__.from_config(layer_config)

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    model = keras.models.Sequential()
    model.add(layer)
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(computed_output_shape,
                                        actual_output_shape):
        if expected_dim is not None:
            if expected_dim != actual_dim:
                raise AssertionError(
                    'When testing layer %s **after deserialization**, '
                    'for input %s, found output_shape='
                    '%s but expected to find inferred shape %s.\nFull kwargs: %s' %
                    (layer_cls.__name__,
                     x,
                     actual_output_shape,
                     computed_output_shape,
                     kwargs))
    if expected_output is not None:
        np.testing.assert_allclose(actual_output, expected_output,
                                   rtol=1e-3, atol=1e-6)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = keras.models.Sequential.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        np.testing.assert_allclose(output, actual_output, rtol=1e-3, atol=1e-6)

    # for further checks in the caller function
    return actual_output

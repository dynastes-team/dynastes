import tensorflow as tf

from tensorflow.python.keras import backend as tf_keras_backend


def image_data_format():
    """Returns the default image data format convention.
    # Returns
        A string, either `'channels_first'` or `'channels_last'`
    # Example
    ```python
        >>> keras.backend.image_data_format()
        'channels_first'
    ```
    """
    return tf_keras_backend.image_data_format()


def normalize_data_format(value):
    """Checks that the value correspond to a valid data format.
    # Arguments
        value: String or None. `'channels_first'` or `'channels_last'`.
    # Returns
        A string, either `'channels_first'` or `'channels_last'`
    # Example
    ```python
        >>> from keras import backend as K
        >>> K.normalize_data_format(None)
        'channels_first'
        >>> K.normalize_data_format('channels_last')
        'channels_last'
    ```
    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    """
    if value is None:
        value = image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format


def transpose_shape(shape, target_format, spatial_axes):
    """Converts a tuple or a list to the correct `data_format`.
    It does so by switching the positions of its elements.
    # Arguments
        shape: Tuple or list, often representing shape,
            corresponding to `'channels_last'`.
        target_format: A string, either `'channels_first'` or `'channels_last'`.
        spatial_axes: A tuple of integers.
            Correspond to the indexes of the spatial axes.
            For example, if you pass a shape
            representing (batch_size, timesteps, rows, cols, channels),
            then `spatial_axes=(2, 3)`.
    # Returns
        A tuple or list, with the elements permuted according
        to `target_format`.
    # Example
    ```python
        >>> from keras.utils.generic_utils import transpose_shape
        >>> transpose_shape((16, 128, 128, 32),'channels_first', spatial_axes=(1, 2))
        (16, 32, 128, 128)
        >>> transpose_shape((16, 128, 128, 32), 'channels_last', spatial_axes=(1, 2))
        (16, 128, 128, 32)
        >>> transpose_shape((128, 128, 32), 'channels_first', spatial_axes=(0, 1))
        (32, 128, 128)
    ```
    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    """
    if target_format == 'channels_first':
        new_values = shape[:spatial_axes[0]]
        new_values += (shape[-1],)
        new_values += tuple(shape[x] for x in spatial_axes)

        if isinstance(shape, list):
            return list(new_values)
        return new_values
    elif target_format == 'channels_last':
        return shape
    else:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(target_format))


def _preprocess_padding(padding):
    """Convert keras' padding to tensorflow's padding.
    # Arguments
        padding: string, `"same"` or `"valid"`.
    # Returns
        a string, `"SAME"` or `"VALID"`.
    # Raises
        ValueError: if `padding` is invalid.
    """
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid padding: ' + str(padding))
    return padding


def _padding_1d(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
    # Returns
        A padded 3D tensor.
    """
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.pad(x, pattern)


def _padding_2d(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A padded 4D tensor.
    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    data_format = normalize_data_format(data_format)

    pattern = [[0, 0],
               list(padding[0]),
               list(padding[1]),
               [0, 0]]
    pattern = transpose_shape(pattern, data_format, spatial_axes=(1, 2))
    return tf.pad(x, pattern)


def pad_input_1d(x, padding, kernel_size, dilation_rate):
    if padding == 'causal':
        left_pad = dilation_rate * (kernel_size - 1)
        x = _padding_1d(x, (left_pad, 0))
        padding = 'valid'
    padding = _preprocess_padding(padding)
    return x, padding


def pad_input_2d(x, padding, kernel_size, dilation_rate=(0, 0)):
    if padding == 'causal':
        left_pad = dilation_rate[0] * (kernel_size[0] - 1)
        x = _padding_2d(x, (left_pad, 0), (0, 0))
        padding = 'valid'
    padding = _preprocess_padding(padding)
    return x, padding

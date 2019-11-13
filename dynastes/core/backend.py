from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.backend import _preprocess_padding, _preprocess_conv2d_input, shape, image_data_format
from tensorflow.python.ops import array_ops

from dynastes.core import nn as dnn


def depthwise_conv2d_transpose(x,
                               kernel,
                               output_shape,
                               strides=(1, 1),
                               padding='valid',
                               data_format=None,
                               dilation_rate=(1, 1)):
    """2D depthwise deconvolution (i.e.
    transposed depthwise convolution).
    Arguments:
        x: Tensor or variable.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: Tuple of 2 integers.
    Returns:
        A tensor, result of transposed 2D convolution.
    Raises:
        ValueError: if `data_format` is neither `channels_last` or
        `channels_first`.
    """

    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ' + str(data_format))

    # `atrous_conv2d_transpose` only supports NHWC format, even on GPU.
    if data_format == 'channels_first' and dilation_rate != (1, 1):
        force_transpose = True
    else:
        force_transpose = False

    x, tf_data_format = _preprocess_conv2d_input(x, data_format, force_transpose)

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        output_shape = (output_shape[0], output_shape[2], output_shape[3],
                        output_shape[1])
    if output_shape[0] is None:
        output_shape = (shape(x)[0],) + tuple(output_shape[1:])

    if isinstance(output_shape, (tuple, list)):
        output_shape = array_ops.stack(list(output_shape))

    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    x = dnn.depthwise_conv2d_transpose(x, kernel, output_shape, strides,
                                       padding=padding,
                                       data_format=tf_data_format)

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x

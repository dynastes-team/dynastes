import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.util.compat import collections_abc


def _get_sequence(value, n, channel_index, name):
    """Formats a value input for gen_nn_ops."""
    if value is None:
        value = [1]
    elif not isinstance(value, collections_abc.Sized):
        value = [value]

    current_n = len(value)
    if current_n == n + 2:
        return value
    elif current_n == 1:
        value = list((value[0],) * n)
    elif current_n == n:
        value = list(value)
    else:
        raise ValueError("{} should be of length 1, {} or {} but was {}".format(
            name, n, n + 2, current_n))

    if channel_index == 1:
        return [1, 1] + value
    else:
        return [1] + value + [1]


def depthwise_conv2d_transpose(
        input,  # pylint: disable=redefined-builtin
        filters,  # pylint: disable=redefined-builtin
        output_shape,
        strides,
        padding="SAME",
        data_format="NHWC",
        dilations=None,
        name=None):
    """The transpose of `depthwise_conv2d`.
    This operation is sometimes called "deconvolution" after [Deconvolutional
    Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf), but is
    actually the transpose (gradient) of `depthwise_conv2d` rather than an actual
    deconvolution.
    Args:
      input: A 4-D `Tensor` of type `float` and shape `[batch, height, width,
        in_channels]` for `NHWC` data format or `[batch, in_channels, height,
        width]` for `NCHW` data format.
      filters: A 4-D `Tensor` with the same type as `input` and shape `[height,
        width, channel_divider, in_channels]`.  `filter`'s `in_channels` dimension
        must match that of `input`.
      output_shape: A 1-D `Tensor` representing the output shape of the
        deconvolution op.
      strides: An int or list of `ints` that has length `1`, `2` or `4`.  The
        stride of the sliding window for each dimension of `input`. If a single
        value is given it is replicated in the `H` and `W` dimension. By default
        the `N` and `C` dimensions are set to 0. The dimension order is determined
        by the value of `data_format`, see below for details.
      padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
        the "returns" section of `tf.nn.convolution` for details.
      data_format: A string. 'NHWC' and 'NCHW' are supported.
      dilations: An int or list of `ints` that has length `1`, `2` or `4`,
        defaults to 1. The dilation factor for each dimension of`input`. If a
        single value is given it is replicated in the `H` and `W` dimension. By
        default the `N` and `C` dimensions are set to 1. If set to k > 1, there
        will be k-1 skipped cells between each filter element on that dimension.
        The dimension order is determined by the value of `data_format`, see above
        for details. Dilations in the batch and depth dimensions if a 4-d tensor
        must be 1.
      name: Optional name for the returned tensor.
    Returns:
      A `Tensor` with the same type as `input`.
    Raises:
      ValueError: If input/output depth does not match `filter`'s shape, or if
        padding is other than `'VALID'` or `'SAME'`.
    """
    with ops.name_scope(name, "depthwise_conv2d_transpose",
                        [input, filter, output_shape]) as name:
        if data_format is None:
            data_format = "NHWC"
        channel_index = 1 if data_format.startswith("NC") else 3

        strides = _get_sequence(strides, 2, channel_index, "strides")
        dilations = _get_sequence(dilations, 2, channel_index, "dilations")

        return tf.nn.depthwise_conv2d_backprop_input(
            input_sizes=output_shape,
            filter=filters,
            out_backprop=input,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilations=dilations,
            name=name)

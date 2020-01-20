import abc

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

from dynastes import regularizers
from dynastes.layers.base_layers import DynastesBaseLayer, ActivatedKernelBiasBaseLayer
from dynastes.ops.t2t_common import shape_list
from dynastes.ops import scale_ops


class _Conv(ActivatedKernelBiasBaseLayer, abc.ABC):
    """Abstract N-D convolution layer (private, used as implementation base).
    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    Arguments:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` the weights of this layer will be marked as
        trainable (and listed in `layer.trainable_weights`).
      name: A string, the name of the layer.
    """

    def __init__(self, rank: int,
                 filters: object,
                 kernel_size: object,
                 strides: object = 1,
                 padding: object = 'valid',
                 dilation_rate: object = 1,
                 activity_regularizer: object = None,
                 input_spec=None,
                 trainable: object = True,
                 name: object = None,
                 **kwargs):
        super(_Conv, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format('channels_last')
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        if input_spec is None:
            input_spec = InputSpec(ndim=self.rank + 2)
        self.input_spec = input_spec

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.build_kernel(kernel_shape)
        self.bias = self.build_bias(self.filters)
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=op_padding,
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       self.rank + 2))
        self._mask_convolution_op = nn_ops.Convolution(
            input_shape[:-1] + [1],
            filter_shape=self.kernel.shape[:-2] + (1, 1),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=op_padding,
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       self.rank + 2))
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask_kernel = self.get_weight('kernel', training=None)
            mask_shapes = max(shape_list(mask_kernel[:-2]))
            if mask_shapes > 1:
                mask = tf.cast(mask, tf.float16)
                mask = tf.expand_dims(mask, axis=-1)
                mask_kernel = tf.reduce_max(tf.abs(mask_kernel), axis=-1, keepdims=True)
                mask_kernel = tf.reduce_max(tf.abs(mask_kernel), axis=-2, keepdims=True)
                mask_kernel = tf.cast(mask_kernel, tf.float16)
                mask = self._mask_convolution_op(mask, mask_kernel)
                mask = mask >= self.mask_threshold
                mask = tf.squeeze(mask, axis=-1)
            return mask
        return None

    def call(self, inputs, training=None, mask=None):
        outputs = self._convolution_op(inputs, self.get_weight('kernel', training=training))
        return super(_Conv, self).call(outputs, training=training)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            [self.filters])
        else:
            raise ValueError('No support for NCHW yet')

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
        }
        base_config = super(_Conv, self).get_config()
        return {**base_config, **config}

    def _compute_causal_padding(self):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)

        if self.data_format == 'channels_last':
            if self.rank == 1:
                causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
            elif self.rank == 2:
                causal_padding = [[0, 0], [left_pad, 0], [0, 0], [0, 0]]
            elif self.rank == 3:
                causal_padding = [[0, 0], [left_pad, 0], [0, 0], [0, 0], [0, 0]]
            else:
                raise ValueError()
            return causal_padding
        else:
            raise ValueError('No support for NCHW yet')


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DynastesConv1D(_Conv):
    """1D convolution layer (e.g. temporal convolution).
    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.
    Arguments:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer,
        specifying the length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
        `"causal"` results in causal (dilated) convolutions, e.g. output[t]
        does not depend on input[t+1:]. Useful when modeling temporal data
        where the model should not violate the temporal order.
        See [WaveNet: A Generative Model for Raw Audio, section
          2.1](https://arxiv.org/abs/1609.03499).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
      dilation_rate: an integer or tuple/list of a single integer, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
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
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    Examples:
      ```python
      # Small convolutional model for 128-length vectors with 6 timesteps
      # model.input_shape == (None, 6, 128)

      model = Sequential()
      model.add(Conv1D(32, 3,
                activation='relu',
                input_shape=(6, 128)))

      # now: model.output_shape == (None, 4, 32)
      ```
    Input shape:
      3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
      3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self,
                 filters: object,
                 kernel_size: object,
                 strides: object = 1,
                 padding: object = 'valid',
                 dilation_rate: object = 1,
                 **kwargs):
        super(DynastesConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            **kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.padding == 'causal':
                mask = tf.cast(mask, inputs.dtype)
                mask = 1 - tf.expand_dims(mask, axis=-1)
                inputs = array_ops.pad(inputs, self._compute_causal_padding())
                mask = array_ops.pad(mask, self._compute_causal_padding())
                mask = (1. - mask) < self.mask_threshold
                mask = tf.squeeze(mask, axis=-1)
        return super(DynastesConv1D, self).compute_mask(inputs, mask)

    def call(self, inputs, training=None, mask=None):
        if self.padding == 'causal':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())
        return super(DynastesConv1D, self).call(inputs, training, mask=mask)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DynastesConv2D(_Conv):
    """2D convolution layer (e.g. spatial convolution over images).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
      dilation_rate: an integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
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
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
      4D tensor with shape:
      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 filters: object,
                 kernel_size: object,
                 strides: object = (1, 1),
                 padding: object = 'valid',
                 dilation_rate: object = (1, 1),
                 **kwargs):
        super(DynastesConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            **kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.padding == 'causal':
                mask = tf.cast(mask, inputs.dtype)
                mask = 1 - tf.expand_dims(mask, axis=-1)
                inputs = array_ops.pad(inputs, self._compute_causal_padding())
                mask = array_ops.pad(mask, self._compute_causal_padding())
                mask = mask >= self.mask_threshold
                mask = tf.squeeze(mask, axis=-1)
        return super(DynastesConv2D, self).compute_mask(inputs, mask)

    def call(self, inputs, training=None, mask=None):
        if self.padding == 'causal':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())
        return super(DynastesConv2D, self).call(inputs, training, mask=mask)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DynastesConv3D(_Conv):
    """2D convolution layer (e.g. spatial convolution over images).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
      dilation_rate: an integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
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
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
      4D tensor with shape:
      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 filters: object,
                 kernel_size: object,
                 strides: object = (1, 1, 1),
                 padding: object = 'valid',
                 dilation_rate: object = (1, 1, 1),
                 **kwargs):
        super(DynastesConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            **kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.padding == 'causal':
                mask = tf.cast(mask, inputs.dtype)
                mask = 1 - tf.expand_dims(mask, axis=-1)
                inputs = array_ops.pad(inputs, self._compute_causal_padding())
                mask = array_ops.pad(mask, self._compute_causal_padding())
                mask = mask >= self.mask_threshold
                mask = tf.squeeze(mask, axis=-1)
        return super(DynastesConv3D, self).compute_mask(inputs, mask)

    def call(self, inputs, training=None, mask=None):
        if self.padding == 'causal':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())
        return super(DynastesConv3D, self).call(inputs, training, mask=mask)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DynastesConv2DTranspose(DynastesConv2D):
    """Transposed convolution layer (sometimes called Deconvolution).
    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    Arguments:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      output_padding: An integer or tuple/list of 2 integers,
        specifying the amount of padding along the height and width
        of the output tensor.
        Can be a single integer to specify the same value for all
        spatial dimensions.
        The amount of output padding along a given dimension must be
        lower than the stride along that same dimension.
        If set to `None` (default), the output shape is inferred.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
      dilation_rate: an integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
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
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    Input shape:
      4D tensor with shape:
      `(batch, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
      4D tensor with shape:
      `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
    References:
      - [A guide to convolution arithmetic for deep
        learning](https://arxiv.org/abs/1603.07285v1)
      - [Deconvolutional
        Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    """

    def __init__(self,
                 filters: object,
                 kernel_size: object,
                 strides: object = (1, 1),
                 padding: object = 'valid',
                 mask_threshold: object = 0.51,
                 output_padding: object = None,
                 dilation_rate: object = (1, 1),
                 **kwargs):
        if 'data_format' in kwargs:
            kwargs.pop('data_format')
        super(DynastesConv2DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            **kwargs)
        self.mask_threshold = mask_threshold
        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                                                     'greater than output padding ' +
                                     str(self.output_padding))

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4. Received input shape: ' +
                             str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.build_kernel(kernel_shape)
        if self.use_bias:
            self.bias = self.build_bias(self.filters)
        else:
            self.bias = None
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:

            mask_kernel = self.get_weight('kernel', training=None)
            mask_shapes = max(shape_list(mask_kernel[:-2]))
            if mask_shapes > 1:
                mask = tf.cast(mask, tf.float16)
                mask = tf.expand_dims(mask, axis=-1)
                mask_kernel = tf.reduce_max(tf.abs(mask_kernel), axis=-1, keepdims=True)
                mask_kernel = tf.reduce_max(tf.abs(mask_kernel), axis=-2, keepdims=True)
                mask_kernel = tf.cast(mask_kernel, tf.float16)

                inputs_shape = array_ops.shape(mask)
                batch_size = inputs_shape[0]
                if self.data_format == 'channels_first':
                    h_axis, w_axis = 2, 3
                else:
                    h_axis, w_axis = 1, 2

                height, width = inputs_shape[h_axis], inputs_shape[w_axis]
                kernel_h, kernel_w = self.kernel_size
                stride_h, stride_w = self.strides

                if self.output_padding is None:
                    out_pad_h = out_pad_w = None
                else:
                    out_pad_h, out_pad_w = self.output_padding

                # Infer the dynamic output shape:
                out_height = conv_utils.deconv_output_length(height,
                                                             kernel_h,
                                                             padding=self.padding,
                                                             output_padding=out_pad_h,
                                                             stride=stride_h,
                                                             dilation=self.dilation_rate[0])
                out_width = conv_utils.deconv_output_length(width,
                                                            kernel_w,
                                                            padding=self.padding,
                                                            output_padding=out_pad_w,
                                                            stride=stride_w,
                                                            dilation=self.dilation_rate[1])
                if self.data_format == 'channels_first':
                    raise ValueError('NCHW not supported yet')
                else:
                    output_shape = (batch_size, out_height, out_width, 1)

                output_shape_tensor = array_ops.stack(output_shape, name=self.name + '_stack_mask_shape_op')
                outputs = backend.conv2d_transpose(
                    mask,
                    mask_kernel,
                    output_shape_tensor,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)

                if not context.executing_eagerly():
                    # Infer the static output shape:
                    out_shape = self._compute_output_shape(mask.shape, mask=True)
                    outputs.set_shape(out_shape)
                mask = outputs > self.mask_threshold
                mask = tf.squeeze(mask, axis=-1)

        return mask

    def call(self, inputs, training=None, mask=None):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(height,
                                                     kernel_h,
                                                     padding=self.padding,
                                                     output_padding=out_pad_h,
                                                     stride=stride_h,
                                                     dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = array_ops.stack(output_shape)
        outputs = backend.conv2d_transpose(
            inputs,
            self.get_weight('kernel', training=training),
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self._compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        return super(DynastesConv2DTranspose, self).post_process_call(outputs, training=training)

    def _compute_output_shape(self, input_shape, mask=False):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        c_axis_shape = self.filters
        if mask:
            c_axis_shape = 1
        output_shape[c_axis] = c_axis_shape
        output_shape[h_axis] = conv_utils.deconv_output_length(
            output_shape[h_axis],
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0])
        output_shape[w_axis] = conv_utils.deconv_output_length(
            output_shape[w_axis],
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1])
        return tensor_shape.TensorShape(output_shape)

    def compute_output_shape(self, input_shape):
        return self._compute_output_shape(input_shape)

    def get_config(self):
        config = super(DynastesConv2DTranspose, self).get_config()
        config['output_padding'] = self.output_padding
        return config


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DynastesDepthwiseConv2D(DynastesConv2D):
    """Depthwise separable 2D convolution.
    Depthwise Separable convolutions consists in performing
    just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.
    Arguments:
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `'valid'` or `'same'` (case-insensitive).
      depth_multiplier: The number of depthwise convolution output channels
        for each input channel.
        The total number of depthwise convolution output
        channels will be equal to `filters_in * depth_multiplier`.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be 'channels_last'.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. 'linear' activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the depthwise kernel matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the depthwise kernel matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its 'activation').
      kernel_constraint: Constraint function applied to
        the depthwise kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
    Input shape:
      4D tensor with shape:
      `[batch, channels, rows, cols]` if data_format='channels_first'
      or 4D tensor with shape:
      `[batch, rows, cols, channels]` if data_format='channels_last'.
    Output shape:
      4D tensor with shape:
      `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
      or 4D tensor with shape:
      `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 kernel_size: object,
                 strides: object = (1, 1),
                 padding: object = 'valid',
                 depth_multiplier: int = 1,
                 **kwargs):
        if 'filters' in kwargs:
            kwargs.pop('filters')
        super(DynastesDepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            **kwargs)
        self.depth_multiplier = depth_multiplier

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.build_kernel(depthwise_kernel_shape)

        if self.use_bias:
            self.bias = self.build_bias(input_dim * self.depth_multiplier)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)
            mask = 1 - tf.expand_dims(mask, axis=-1)
            padding = self.padding
            if self.padding == 'causal':
                mask = array_ops.pad(mask, self._compute_causal_padding())
                padding = 'valid'
            mask_kernel = self.get_weight('kernel', training=None)
            mask_shapes = max(shape_list(mask_kernel[:-2]))
            if mask_shapes > 1:
                mask_kernel = tf.reduce_max(tf.abs(mask_kernel), axis=-2, keepdims=True)
                mask_kernel = tf.cast(mask_kernel, inputs.dtype)
                mask = backend.depthwise_conv2d(
                    mask,
                    mask_kernel,
                    strides=self.strides,
                    padding=padding,
                    dilation_rate=self.dilation_rate,
                    data_format=self.data_format)
                mask = mask >= self.mask_threshold
                mask = tf.squeeze(mask, axis=-1)
                mask = tf.cast(mask, tf.bool)
            return mask
        return mask

    def call(self, inputs, training=None, mask=None):
        padding = self.padding
        if padding == 'causal':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())
            padding = 'valid'
        outputs = backend.depthwise_conv2d(
            inputs,
            self.get_weight("kernel", training=training),
            strides=self.strides,
            padding=padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        return super(DynastesDepthwiseConv2D, self).post_process_call(outputs, training=training)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            raise ValueError('No support for NCHW yet')
        rows = input_shape[1]
        cols = input_shape[2]
        out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])
        if self.data_format == 'channels_first':
            return input_shape[0], out_filters, rows, cols
        elif self.data_format == 'channels_last':
            return input_shape[0], rows, cols, out_filters

    def get_config(self):
        config = super(DynastesDepthwiseConv2D, self).get_config()
        config.pop('filters')
        config['depth_multiplier'] = self.depth_multiplier
        return config


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DynastesDepthwiseConv1D(DynastesDepthwiseConv2D):
    def __init__(self,
                 kernel_size: int = 1,
                 strides: int = 1,
                 padding='valid',
                 depth_multiplier: int = 1,
                 dilation_rate: int = 1,
                 **kwargs):
        kwargs['input_spec'] = InputSpec(ndim=3)
        super(DynastesDepthwiseConv1D, self).__init__(
            filters=None,
            kernel_size=(kernel_size, 1),
            strides=(strides, strides),
            padding=padding,
            depth_multiplier=depth_multiplier,
            dilation_rate=(dilation_rate, 1),
            **kwargs)

    def build(self, input_shape):
        input_shape = input_shape[:-1] + [1, input_shape[-1]]
        super().build(input_shape)
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            raise ValueError('No support for NCHW yet, maybe ever')
        else:
            channel_axis = 3
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv1D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=3, axes={2: input_dim})

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.expand_dims(inputs, axis=-2)
            mask = tf.expand_dims(mask, axis=-1)
            mask = super().compute_mask(inputs, mask)
            if mask is not None:
                mask = tf.squeeze(mask, axis=-1)
            return mask
        return None

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-2)
        if mask is not None:
            mask = tf.expand_dims(inputs, axis=-1)
        outputs = super().call(inputs, training, mask)
        return tf.squeeze(outputs, axis=-2)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[:-1] + [1, input_shape[-1]]
        output_shape = super().compute_output_shape(input_shape)
        return output_shape[:-2] + (output_shape[-1],)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size[0],
            'strides': self.strides[0],
            'padding': self.padding,
            'dilation_rate': self.dilation_rate[0],
        }
        base_config = super(DynastesDepthwiseConv1D, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class DynastesConv1DTranspose(DynastesConv2DTranspose):
    def __init__(self,
                 filters,
                 kernel_size: int = 1,
                 strides: int = 1,
                 padding='valid',
                 dilation_rate: int = 1,
                 **kwargs):
        kwargs['input_spec'] = InputSpec(ndim=3)
        super(DynastesConv1DTranspose, self).__init__(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding=padding,
            dilation_rate=(dilation_rate, 1),
            **kwargs)

    def build(self, input_shape):
        input_shape = input_shape[:-1] + [1, input_shape[-1]]
        super().build(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.expand_dims(inputs, axis=-2)
            mask = tf.expand_dims(mask, axis=-1)
            mask = super().compute_mask(inputs, mask=mask)
            if mask is not None:
                mask = tf.squeeze(mask, axis=-1)
            return mask
        return None

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-2)
        if mask is not None:
            mask = tf.expand_dims(inputs, axis=-1)
        outputs = super().call(inputs, training, mask)
        return tf.squeeze(outputs, axis=-2)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[:-1] + [1, input_shape[-1]]
        output_shape = super().compute_output_shape(input_shape)
        return output_shape[:-2] + (output_shape[-1],)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size[0],
            'strides': self.strides[0],
            'padding': self.padding,
            'dilation_rate': self.dilation_rate[0],
        }
        base_config = super(DynastesConv1DTranspose, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class Upsampling2D(DynastesBaseLayer):

    def __init__(self,
                 strides=(2, 2),
                 method='bilinear',
                 **kwargs):
        super(Upsampling2D, self).__init__(**kwargs)
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.method = method

    def _resize(self, x):
        return scale_ops.upscale2d(x, strides=self.strides, method=self.method)

    def _resize_cheap(self, x):
        return scale_ops.upscale2d(x, strides=self.strides, method='nearest', antialias=False)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float16)
            mask = tf.expand_dims(mask, axis=-1)

            mask = self._resize_cheap(mask)
            mask = tf.squeeze(mask, axis=-1)
            mask = mask >= self.mask_threshold
        return mask

    def call(self, inputs, training=None, mask=None):
        return self._resize(inputs)

    def compute_output_shape(self, input_shape):
        out_shape = input_shape[0], input_shape[1] * self.strides[0], input_shape[2] * self.strides[1], input_shape[3]
        return tensor_shape.TensorShape(out_shape)

    def get_config(self):
        config = {
            'strides': self.strides,
            'method': self.method,
        }
        base_config = super(Upsampling2D, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class Upsampling1D(Upsampling2D):

    def __init__(self,
                 strides=2,
                 method='bilinear',
                 **kwargs):
        super(Upsampling1D, self).__init__(strides=(strides, 1), method=method, **kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)
            mask = super(Upsampling1D, self).compute_mask(inputs=None, mask=mask)
            mask = tf.squeeze(mask, axis=-1)
        return mask

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-2)
        x = super(Upsampling1D, self).call(inputs, training=training, mask=mask)
        return tf.squeeze(x, axis=-2)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[:-1] + [1, input_shape[-1]]
        output_shape = super().compute_output_shape(input_shape)
        return output_shape[:-2] + (output_shape[-1],)

    def get_config(self):
        config = {
            'strides': self.strides[0],
            'method': self.method,
        }
        base_config = super(Upsampling1D, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class Downsampling2D(DynastesBaseLayer):

    def __init__(self,
                 strides=(2, 2),
                 method='bilinear',
                 **kwargs):
        super(Downsampling2D, self).__init__(**kwargs)
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.method = method


    def _resize(self, x):
        return scale_ops.downscale2d(x, strides=self.strides, method=self.method)

    def _resize_cheap(self, x):
        return scale_ops.downscale2d(x, strides=self.strides, method='nearest', antialias=False)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float16)
            mask = tf.expand_dims(mask, axis=-1)

            mask = self._resize_cheap(mask)
            mask = tf.squeeze(mask, axis=-1)
            mask = mask >= self.mask_threshold
        return mask

    def call(self, inputs, training=None, mask=None):
        return self._resize(inputs)

    def compute_output_shape(self, input_shape):
        out_shape = input_shape[0], input_shape[1] // self.strides[0], input_shape[2] // self.strides[1], input_shape[3]
        return tensor_shape.TensorShape(out_shape)

    def get_config(self):
        config = {
            'strides': self.strides,
            'method': self.method,
        }
        base_config = super(Downsampling2D, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class Downsampling1D(Downsampling2D):

    def __init__(self,
                 strides=2,
                 method='bilinear',
                 **kwargs):
        super(Downsampling1D, self).__init__(strides=(strides, 1), method=method, **kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)
            mask = super(Downsampling1D, self).compute_mask(inputs=None, mask=mask)
            mask = tf.squeeze(mask, axis=-1)
        return mask

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-2)
        x = super(Downsampling1D, self).call(inputs, training=training, mask=mask)
        return tf.squeeze(x, axis=-2)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[:-1] + [1, input_shape[-1]]
        output_shape = super().compute_output_shape(input_shape)
        return output_shape[:-2] + (output_shape[-1],)

    def get_config(self):
        config = {
            'strides': self.strides[0],
            'method': self.method,
        }
        base_config = super(Downsampling1D, self).get_config()
        return {**base_config, **config}


# Export aliases
Conv1D = DynastesConv1D
Conv2D = DynastesConv2D
Conv3D = DynastesConv3D
Conv1DTranspose = DynastesConv1DTranspose
Conv2DTranspose = DynastesConv2DTranspose
DepthwiseConv1D = DynastesDepthwiseConv1D
DepthwiseConv2D = DynastesDepthwiseConv2D

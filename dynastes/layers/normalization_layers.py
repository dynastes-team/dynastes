import abc

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables

tfal = None
try:
    import tensorflow_addons.layers as tfal
except:
    print('WARNING! TensorFlow Addons are missing!')
    tfal = None

from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.layers.conditioning_layers import FeaturewiseLinearModulation, ModulationLayer
from dynastes.ops.t2t_common import shape_list


def masked_moments(x, axes, mask=None, keepdims=False, epsilon=1e-15):
    if mask is None:
        return tf.nn.moments(x, axes=axes, keepdims=keepdims)
    else:
        x_shape = shape_list(x)
        mask_shape = shape_list(mask)
        _mask = tf.reshape(tf.cast(mask, x.dtype), mask_shape + [1] * (len(x_shape) - len(mask_shape)))
    n_mask_indices = tf.reduce_sum(_mask, axis=axes, keepdims=True)
    _mean = tf.reduce_sum(x, axis=axes, keepdims=True) / tf.cast(
        tf.maximum(tf.cast(1, n_mask_indices.dtype), n_mask_indices), x.dtype)
    var = tf.reduce_sum(tf.math.squared_difference(x, _mean), axis=axes, keepdims=True) / tf.cast(
        tf.maximum(tf.cast(1, n_mask_indices.dtype), n_mask_indices - 1), x.dtype)
    return tf.reduce_sum(_mean, axis=axes, keepdims=keepdims), tf.reduce_sum(var, axis=axes, keepdims=keepdims)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class PoolNormalization2D(DynastesBaseLayer):
    """
    My (Göran Sandström) own invention, performs smooth local response normalization
    using downsampling and upsampling
    """

    def __init__(self,
                 pool_size=(2, 2),
                 method=tf.image.ResizeMethod.BILINEAR, antialias=True,
                 **kwargs):
        super(PoolNormalization2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.method = method
        self.antialias = antialias

    def call(self, x, training=None):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x_size = shape_list(x)[1:-1]
        x_size = np.where(np.array(list(self.pool_size)) == -1, 1, x_size).tolist()
        x_size = tf.convert_to_tensor(x_size)
        reduce_axes = np.where(np.array(self.pool_size) == -1)[0].tolist()
        if len(reduce_axes) == 2:
            x -= tf.reduce_mean(x, axis=reduce_axes, keepdims=True)
            x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=reduce_axes, keepdims=True) + 1e-8)
            x = tf.cast(x, orig_dtype)
            return x
        pool_size_t = tf.convert_to_tensor(self.pool_size)
        pool_size_t = tf.maximum(pool_size_t, 1)
        pooled_size = x_size // pool_size_t

        def pool_reduce(x, dtype=tf.float32):
            if len(reduce_axes) > 0:
                x = tf.reduce_mean(x, axis=reduce_axes, keepdims=True)
            x = tf.cast(tf.image.resize(tf.image.resize(x,
                                                        pooled_size,
                                                        method=self.method,
                                                        antialias=self.antialias),
                                        x_size,
                                        method=self.method,
                                        antialias=self.antialias), dtype)
            return x

        x -= pool_reduce(x, tf.float32)
        x *= tf.math.rsqrt(pool_reduce(tf.square(x), tf.float32) + 1e-8)
        x = tf.cast(x, orig_dtype)
        return x

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'method': self.method,
            'antialias': self.antialias,
        }
        base_config = super(PoolNormalization2D, self).get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class BatchNormalization(DynastesBaseLayer):
    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-8,
                 axis=-1,
                 scale=True,
                 center=True,
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 **kwargs):
        kwargs['moving_mean_initializer'] = moving_mean_initializer
        kwargs['moving_variance_initializer'] = moving_variance_initializer
        super(BatchNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.axis = axis
        self.scale = scale
        self.center = center
        self.momentum = momentum
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def build(self, input_shape):
        param_shape = input_shape[-1]
        if self.center:
            self.beta = self.add_weight('beta', shape=[param_shape])
        if self.scale:
            self.gamma = self.add_weight('gamma', shape=[param_shape])

        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=param_shape,
            dtype=self.dtype,
            synchronization=tf_variables.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.MEAN,
            experimental_autocast=False)

        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=param_shape,
            dtype=self.dtype,
            synchronization=tf_variables.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.MEAN,
            experimental_autocast=False)

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
                if decay.dtype != variable.dtype.base_dtype:
                    decay = math_ops.cast(decay, variable.dtype.base_dtype)
                update_delta = (
                                       variable - math_ops.cast(value, variable.dtype)) * decay
                if inputs_size is not None:
                    update_delta = array_ops.where(inputs_size > 0, update_delta,
                                                   K.zeros_like(update_delta))
                return state_ops.assign_sub(variable, update_delta, name=scope)

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        training = math_ops.logical_and(training, self.trainable)
        return training

    def call(self, inputs, training=None, mask=None, **kwargs):
        training = self._get_training_value(training)
        x = inputs
        if mask is not None:
            x = tf.where(tf.expand_dims(mask, axis=-1), x, tf.zeros_like(x))
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        inputs_size = array_ops.size(inputs)
        axes = list(range(len(shape_list(x))))[:-1]
        training_value = tf_utils.constant_value(training)
        if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
            mean, variance = self.moving_mean, self.moving_variance
        else:
            mean, variance = masked_moments(x, mask=mask, axes=axes, keepdims=False)
            mean = tf.squeeze(mean)
            variance = tf.squeeze(variance)
            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = tf_utils.smart_cond(training,
                                       lambda: mean,
                                       lambda: ops.convert_to_tensor(moving_mean))
            variance = tf_utils.smart_cond(
                training,
                lambda: variance,
                lambda: tf.convert_to_tensor(moving_variance))

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum,
                                                   inputs_size)

            def mean_update():
                true_branch = lambda: _do_update(self.moving_mean, mean)
                false_branch = lambda: self.moving_mean
                return tf_utils.smart_cond(training, true_branch, false_branch)

            def variance_update():
                """Update the moving variance."""

                true_branch = lambda: _do_update(self.moving_variance, variance)

                false_branch = lambda: self.moving_variance
                return tf_utils.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        if self.scale:
            gamma = self.get_weight('gamma', training=training)
        else:
            gamma = None
        if self.center:
            beta = self.get_weight('beta', training=training)
        else:
            beta = None

        x = tf.nn.batch_normalization(
            x,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon)
        x = tf.cast(x, orig_dtype)
        return x

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'scale': self.scale,
            'center': self.center,
            'momentum': self.momentum
        }
        base_config = super(BatchNormalization, self).get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class LayerNormalization(DynastesBaseLayer):
    def __init__(self,
                 epsilon=1e-8,
                 axis=-1,
                 scale=True,
                 center=True,
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.axis = axis
        self.scale = scale
        self.center = center

    def build(self, input_shape):
        if self.center:
            self.beta = self.add_weight('beta', shape=[input_shape[-1]])
        if self.scale:
            self.gamma = self.add_weight('gamma', shape=[input_shape[-1]])

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, training=None, mask=None, **kwargs):
        x = inputs
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)

        mean, var = masked_moments(x, mask=mask, axes=self.axis, keepdims=True)
        if self.scale:
            gamma = self.get_weight('gamma', training=training)
        else:
            gamma = None
        if self.center:
            beta = self.get_weight('beta', training=training)
        else:
            beta = None

        x = tf.nn.batch_normalization(
            x,
            mean=mean,
            variance=var,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon)
        x = tf.cast(x, orig_dtype)
        return x

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'scale': self.scale,
            'center': self.center
        }
        base_config = super(LayerNormalization, self).get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class InstanceNormalization(DynastesBaseLayer):
    def __init__(self,
                 epsilon=1e-8,
                 axes=(1, 2),
                 scale=True,
                 center=True,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.axes = axes
        self.scale = scale
        self.center = center

    def build(self, input_shape):
        if self.center:
            self.beta = self.add_weight('beta', shape=[input_shape[-1]])
        if self.scale:
            self.gamma = self.add_weight('gamma', shape=[input_shape[-1]])

    def call(self, inputs, training=None, mask=None, **kwargs):
        x = inputs
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)

        mean, var = masked_moments(x, mask=mask, axes=self.axes, keepdims=True)
        if self.scale:
            gamma = self.get_weight('gamma', training=training)
        else:
            gamma = None
        if self.center:
            beta = self.get_weight('beta', training=training)
        else:
            beta = None

        x = tf.nn.batch_normalization(
            x,
            mean=mean,
            variance=var,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon)
        x = tf.cast(x, orig_dtype)
        return x

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'axes': self.axes,
            'scale': self.scale,
            'center': self.center
        }
        base_config = super(InstanceNormalization, self).get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class InstanceNormalization2D(InstanceNormalization):

    def __init__(self,
                 epsilon=1e-8,
                 **kwargs):
        kwargs['axes'] = (1, 2)
        super(InstanceNormalization2D, self).__init__(epsilon=epsilon, **kwargs)

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
        }
        base_config = super(InstanceNormalization2D, self).get_config()
        f_config = {**base_config, **config}
        f_config.pop('axes')
        return f_config


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class InstanceNormalization1D(InstanceNormalization):

    def __init__(self,
                 epsilon=1e-8,
                 **kwargs):
        kwargs['axes'] = (1,)
        super(InstanceNormalization1D, self).__init__(epsilon=epsilon, **kwargs)

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
        }
        base_config = super(InstanceNormalization1D, self).get_config()
        f_config = {**base_config, **config}
        f_config.pop('axes')
        return f_config


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class MultiNormalization(DynastesBaseLayer):
    def __init__(self,
                 layers,
                 **kwargs):
        super(MultiNormalization, self).__init__(**kwargs)
        self.norm_layers = layers

    def build(self, input_shape):
        for layer in self.norm_layers:
            layer.build(input_shape)
        self.add_weight('balance', [input_shape[-1], len(self.norm_layers)], trainable=True,
                        constraint=lambda x: tf.math.softmax(x, axis=-1))
        super(MultiNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        x = tf.stack([layer(inputs, training=training) for layer in self.norm_layers], axis=-1)
        x *= self.get_weight('balance', training=training)
        x = tf.reduce_sum(x, axis=-1)
        return x


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class ModulatedNormalization(DynastesBaseLayer, abc.ABC):

    def __init__(self, modulation_layer: ModulationLayer,
                 norm_layer, **kwargs):
        super(ModulatedNormalization, self).__init__(**kwargs)
        self.modulation_layer = modulation_layer
        self.norm_layer = norm_layer

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.modulation_layer.build(input_shape)
        self.norm_layer.build(input_shape[0])
        super(ModulatedNormalization, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if type(mask) == list:
                return mask[0]
            return mask

    def call(self, inputs, training=None):
        orig_dtype = inputs[0].dtype
        normalized = self.norm_layer(inputs[0], training=training)
        return tf.cast(self.modulation_layer([normalized] + inputs[1:], training=training), orig_dtype)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class AdaptiveNormalization(ModulatedNormalization, abc.ABC):

    def __init__(self, norm_layer, method=tf.image.ResizeMethod.BILINEAR, antialias=True, **kwargs):
        super(AdaptiveNormalization, self).__init__(
            modulation_layer=FeaturewiseLinearModulation(method=method, antialias=antialias),
            norm_layer=norm_layer,
            **kwargs)

    def get_config(self):
        config = {
            'method': self.method,
            'antialias': self.antialias,
            'mode': self.mode,
        }
        base_config = super(AdaptiveNormalization, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class AdaptiveGroupNormalization(AdaptiveNormalization):

    def __init__(self, n_groups=2,
                 **kwargs):
        super(AdaptiveGroupNormalization, self).__init__(
            tfal.normalizations.GroupNormalization(groups=n_groups, center=False, scale=False), **kwargs)
        self.n_groups = n_groups

    def get_config(self):
        config = {
            'n_groups': self.n_groups,
        }
        base_config = super(AdaptiveGroupNormalization, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class AdaptiveInstanceNormalization(AdaptiveNormalization):
    """
    Introduced in:
    Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
    https://arxiv.org/abs/1703.06868

    Used in (notable):
    A Style-Based Generator Architecture for Generative Adversarial Networks ("StyleGAN")
    https://arxiv.org/abs/1812.04948

    Can also perform "Spatially-Adaptive Normalization":
    Semantic Image Synthesis with Spatially-Adaptive Normalization
    https://arxiv.org/abs/1903.07291
    """

    def __init__(self,
                 axes=(1, 2),
                 epsilon=1e-8,
                 **kwargs):
        super(AdaptiveInstanceNormalization, self).__init__(
            LayerNormalization(axes=axes, epsilon=epsilon), **kwargs)
        self.epsilon = epsilon
        self.axes = axes

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'axes': self.axes,
        }
        base_config = super(AdaptiveInstanceNormalization, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class AdaptiveLayerNormalization(AdaptiveGroupNormalization):

    def __init__(self,
                 **kwargs):
        kwargs["n_groups"] = 1
        super(AdaptiveLayerNormalization, self).__init__(**kwargs)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class AdaptiveMultiNormalization(AdaptiveNormalization):

    def __init__(self,
                 layers,
                 **kwargs):
        super(AdaptiveMultiNormalization, self).__init__(MultiNormalization(layers), **kwargs)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class AdaptiveLayerInstanceNormalization(AdaptiveMultiNormalization):
    """
    Introduced in:
    U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation
    https://arxiv.org/abs/1907.10830
    """

    def __init__(self,
                 axes=(1, 2),
                 **kwargs):
        kwargs["layers"] = [LayerNormalization(scale=False, center=False),
                            InstanceNormalization(axes=axes, center=False, scale=False)]
        super(AdaptiveLayerInstanceNormalization, self).__init__(**kwargs)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class AdaptiveLayerInstanceNormalization1D(AdaptiveLayerInstanceNormalization):
    """
    Introduced in:
    U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation
    https://arxiv.org/abs/1907.10830
    """

    def __init__(self,
                 **kwargs):
        kwargs['axes'] = (1,)
        super(AdaptiveLayerInstanceNormalization1D, self).__init__(**kwargs)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class AdaptiveLayerInstanceNormalization2D(AdaptiveLayerInstanceNormalization):
    """
    Introduced in:
    U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation
    https://arxiv.org/abs/1907.10830
    """

    def __init__(self,
                 **kwargs):
        kwargs['axes'] = (1, 2)
        super(AdaptiveLayerInstanceNormalization2D, self).__init__(**kwargs)

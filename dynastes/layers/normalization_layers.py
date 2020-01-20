import abc

import numpy as np
import tensorflow as tf

tfal = None
try:
    import tensorflow_addons.layers as tfal
except:
    print('WARNING! TensorFlow Addons are missing!')
    tfal = None

from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.layers.conditioning_layers import FeaturewiseLinearModulation, ModulationLayer
from dynastes.ops.t2t_common import shape_list


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
class InstanceNormalization(DynastesBaseLayer):
    def __init__(self,
                 epsilon=1e-8,
                 axes=(1, 2),
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.axes = axes

    def call(self, inputs, **kwargs):
        x = inputs
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=self.axes, keepdims=True)
        epsilon = tf.constant(self.epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=self.axes, keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        return x

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'axes': self.axes,
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
            InstanceNormalization(axes=axes, epsilon=epsilon), **kwargs)
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
                 **kwargs):
        kwargs["layers"] = [tfal.normalizations.GroupNormalization(groups=1, center=False, scale=False),
                            tfal.normalizations.GroupNormalization(groups=-1, center=False, scale=False)]
        super(AdaptiveLayerInstanceNormalization, self).__init__(**kwargs)

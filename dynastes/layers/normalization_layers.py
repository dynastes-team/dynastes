import abc

import numpy as np
import tensorflow as tf
import tensorflow_addons.layers as tfal

from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.ops.t2t_common import shape_list


class PoolNormalization2D(DynastesBaseLayer):
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


class _AdaptiveNormalization(DynastesBaseLayer, abc.ABC):

    def __init__(self, method=tf.image.ResizeMethod.BILINEAR, antialias=True, mode=None, **kwargs):
        super(_AdaptiveNormalization, self).__init__(**kwargs)
        self.method = method
        self.antialias = antialias
        self.mode = mode

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        if len(input_shape) == 3:
            assert (input_shape[1][-1] + input_shape[2][-1]) // 2 == input_shape[0][-1]
            if self.mode is not None:
                assert self.mode == 'provided_mean_var'
            else:
                self.mode = 'provided_mean_var'
        elif len(input_shape) == 2:
            if (input_shape[1][-1] // 2) != input_shape[0][-1]:
                if self.mode is not None:
                    assert self.mode == 'mapped'
                else:
                    self.mode = 'mapped'
                self.add_weight('map_kernel', shape=[input_shape[1][-1], input_shape[0][-1] * 2])
            else:
                if self.mode is not None:
                    assert self.mode == 'provided_meanvar_fused'
                else:
                    self.mode = 'provided_meanvar_fused'
        else:
            raise ValueError('Incorrect input shapes')
        super(_AdaptiveNormalization, self).build(input_shape)

    def call_mod(self, x, mean_var, training=None, mask=None):
        mean_var_shape = shape_list(mean_var)
        if self.mode == 'provided_mean_var':
            mean_var = tf.concat([mean_var], axis=-1)
        elif self.mode == 'mapped':
            mean_var = tf.matmul(mean_var[0], self.get_weight('map_kernel', training=training))
        elif self.mode != 'provided_meanvar_fused':
            raise ValueError('Something is wrong')
        mean_var_shape = shape_list(mean_var)
        x_shape = shape_list(x)
        if len(mean_var_shape) != len(x_shape):
            mean_var = tf.reshape(mean_var, [-1, ] + ([1] * (len(x.shape) - 2)) + [2, x.shape[-1]])
        else:
            mean_var_shape_npa = np.asarray(mean_var_shape[1:-1])
            x_shape_npa = np.asarray(x_shape[1:-1])
            compatible = np.all(np.logical_or(mean_var_shape_npa == 1, mean_var_shape_npa == x_shape_npa))
            if not compatible:
                size = np.where(mean_var_shape_npa == 1, mean_var_shape_npa, x_shape_npa).tolist()
                if len(mean_var_shape) == 4:
                    mean_var = tf.image.resize(mean_var, size, method=self.method)
                elif len(mean_var_shape) == 3:
                    mean_var = tf.squeeze(tf.image.resize(tf.expand_dims(mean_var, 1), [1] + size, method=self.method,
                                                          antialias=self.antialias), axis=1)
                else:
                    raise ValueError('Only works for 1D or 2D tensors')
            shape = [mean_var_shape[0]] + np.where(mean_var_shape_npa == 1, mean_var_shape_npa,
                                                   x_shape_npa).tolist() + [2, x.shape[-1]]
            mean_var = tf.reshape(mean_var, shape)
        mean, var = tf.unstack(mean_var, axis=-2)

        return (x * var) + mean

    def get_config(self):
        config = {
            'method': self.method,
            'antialias': self.antialias,
            'mode': self.mode,
        }
        base_config = super(_AdaptiveNormalization, self).get_config()
        return {**base_config, **config}


class AdaptiveNormalization(_AdaptiveNormalization):
    def __init__(self,
                 norm_layer,
                 **kwargs):
        super(AdaptiveNormalization, self).__init__(**kwargs)
        self.norm_layer = norm_layer

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.norm_layer.build(input_shape[0])
        super(AdaptiveNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        orig_dtype = inputs[0].dtype
        normalized = self.norm_layer(inputs[0], training=training)
        return tf.cast(self.call_mod(normalized, mean_var=inputs[1:], training=training), orig_dtype)


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


class AdaptiveInstanceNormalization(AdaptiveGroupNormalization):

    def __init__(self,
                 **kwargs):
        kwargs["groups"] = -1
        super(AdaptiveInstanceNormalization, self).__init__(**kwargs)


class AdaptiveLayerNormalization(AdaptiveGroupNormalization):

    def __init__(self,
                 **kwargs):
        kwargs["groups"] = 1
        super(AdaptiveLayerNormalization, self).__init__(**kwargs)


class AdaptiveMultiNormalization(AdaptiveNormalization):

    def __init__(self,
                 layers,
                 **kwargs):
        super(AdaptiveMultiNormalization, self).__init__(MultiNormalization(layers), **kwargs)


class AdaptiveLayerInstanceNormalization(AdaptiveMultiNormalization):

    def __init__(self,
                 **kwargs):
        kwargs["layers"] = [tfal.normalizations.GroupNormalization(groups=1, center=False, scale=False),
                            tfal.normalizations.GroupNormalization(groups=-1, center=False, scale=False)]
        super(AdaptiveLayerInstanceNormalization, self).__init__(**kwargs)

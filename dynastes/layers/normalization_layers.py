import abc

import tensorflow as tf
import tensorflow_addons.layers as tfal
import numpy as np

from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.ops.t2t_common import shape_list


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

    def __init__(self, method=tf.image.ResizeMethod.BILINEAR, **kwargs):
        super(_AdaptiveNormalization, self).__init__(**kwargs)
        self.method = method
        self.mode = None

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
                    mean_var = tf.squeeze(tf.image.resize(tf.expand_dims(mean_var, 1), [1] + size, method=self.method), axis=1)
                else:
                    raise ValueError('Only works for 1D or 2D tensors')
            shape = [mean_var_shape[0]] + np.where(mean_var_shape_npa == 1, mean_var_shape_npa, x_shape_npa).tolist() + [2, x.shape[-1]]
            mean_var = tf.reshape(mean_var, shape)
        mean, var = tf.unstack(mean_var, axis=-2)

        return (x * var) + mean


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
        normalized = self.norm_layer(inputs[0], training=training)
        return self.call_mod(normalized, mean_var=inputs[1:], training=training)


class AdaptiveGroupNormalization(AdaptiveNormalization):

    def __init__(self, n_groups=2,
                 **kwargs):
        super(AdaptiveGroupNormalization, self).__init__(
            tfal.normalizations.GroupNormalization(groups=n_groups, center=False, scale=False), **kwargs)


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

import abc

import tensorflow as tf
import tensorflow_addons.layers as tfal

from dynastes.layers.base_layers import DynastesBaseLayer


class _AdaptiveNormalization(DynastesBaseLayer, abc.ABC):

    def __init__(self, **kwargs):
        super(_AdaptiveNormalization, self).__init__(**kwargs)
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

        if self.mode == 'provided_mean_var':
            mean = tf.reshape(mean_var[0], [-1] + [1] * (len(x.shape) - 2) + [x.shape[-1]])
            var = tf.reshape(mean_var[1], [-1] + [1] * (len(x.shape) - 2) + [x.shape[-1]])
            return (x * var) + mean
        elif self.mode == 'provided_meanvar_fused':
            mean_var = tf.reshape(mean_var, [-1, ] + ([1] * (len(x.shape) - 2)) + [2, x.shape[1]])
            mean, var = tf.unstack(mean_var, axis=-2)
            return (x * var) + mean
        elif self.mode == 'mapped':
            mean_var = tf.matmul(mean_var, self.get_weight('map_kernel', training=training))
            mean_var = tf.reshape(mean_var, [-1, ] + ([1] * (len(x.shape) - 2)) + [2, x.shape[1]])
            mean, var = tf.unstack(mean_var, axis=-2)
            return (x * var) + mean
        else:
            raise ValueError('Something is wrong')


class AdaptiveGroupNormalization(_AdaptiveNormalization):

    def __init__(self, n_groups=2,
                 **kwargs):
        super(AdaptiveGroupNormalization, self).__init__(**kwargs)
        self.group_normalization = tfal.normalizations.GroupNormalization(groups=n_groups, center=False, scale=False)

    def build(self, input_shape):
        self.group_normalization.build(input_shape[0])
        super(AdaptiveGroupNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        normalized = self.group_normalization(inputs[0], training=training)
        return self.call_mod(normalized, mean_var=inputs[1:], training=training)


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


class AdaptiveLayerInstanceNormalization(_AdaptiveNormalization):

    def __init__(self,
                 **kwargs):
        super(AdaptiveLayerInstanceNormalization, self).__init__(**kwargs)
        self.instance_norm = tfal.normalizations.GroupNormalization(groups=1, center=False, scale=False)
        self.layer_norm = tfal.normalizations.GroupNormalization(groups=-1, center=False, scale=False)

    def build(self, input_shape):
        self.instance_norm.build(input_shape[0])
        self.layer_norm.build(input_shape[0])
        self.add_weight('balance', [input_shape[-1]], trainable=True,
                        initializer=tf.keras.initializers.constant(0.5),
                        constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        super(AdaptiveLayerInstanceNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        instance_normalized = self.instance_norm(inputs[0], training=training)
        layer_normalized = self.layer_norm(inputs[0], training=training)
        balance_weight = self.get_weight('balance', training=training)
        balance_weight = tf.clip_by_value(balance_weight - tf.constant(0.1), 0.0, 1.0)
        normalized = balance_weight * instance_normalized + (1 - balance_weight) * layer_normalized
        return self.call_mod(normalized, mean_var=inputs[1:], training=training)

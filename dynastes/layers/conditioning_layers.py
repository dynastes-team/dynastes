import abc

import numpy as np
import tensorflow as tf

from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.ops.t2t_common import shape_list


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class ModulationLayer(DynastesBaseLayer, abc.ABC):

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class FeaturewiseLinearModulation(ModulationLayer):
    """
        Call accepts list of:
            * [input, modulation]
            * [input, modulation bias, modulation scale]
            * [input, modulation bias + modulation scale]

        Modulation can have smaller dimensions than input,
        as long as '''rank(modulation) in [rank(input), 2]'''
            input: [b, H, W, CH]
                modulation:
                    [b, H, W, N] - Valid
                    [b, N] - Valid
                    [b, other H, other W, N] - Valid
                    [b, any, N] - NOT VALID

        More information
        https://distill.pub/2018/feature-wise-transformations/
    """

    def __init__(self, method=tf.image.ResizeMethod.BILINEAR, antialias=True, mode=None, **kwargs):

        """
        @param method: Method used for scaling conditioning in case rank(input) == rank(modulation)
        @type method: tensorflow.image.ResizeMethod
        @param antialias: Use antialiasing if downscaling conditioning
        @type antialias:
        @param mode: one of 'provided_mean_var', 'mapped' or 'provided_meanvar_fused',
                     leave blank to infer on build
        @type mode: str
        """
        super(FeaturewiseLinearModulation, self).__init__(**kwargs)
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
        super(FeaturewiseLinearModulation, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        assert isinstance(x, list)
        if self.mode == 'provided_mean_var':
            x, mean, var = x
            mean_var = [mean, var]
            mean_var = tf.concat([mean_var], axis=-1)
        elif self.mode == 'mapped':
            x, mean_var = x
            mean_var = tf.matmul(mean_var, self.get_weight('map_kernel', training=training))
        elif self.mode != 'provided_meanvar_fused':
            raise ValueError('Something is wrong')
        else:
            x, mean_var = x
        mean_var_shape = shape_list(mean_var)
        x_shape = shape_list(x)
        if len(mean_var_shape) != len(x_shape):
            mean_var = tf.reshape(mean_var, [-1, ] + ([1] * (len(x.shape) - 2)) + [2, x.shape[-1]])
        else:
            mean_var_shape_npa = np.array(mean_var_shape[1:-1])
            x_shape_npa = np.array(x_shape[1:-1])
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
        base_config = super(FeaturewiseLinearModulation, self).get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]

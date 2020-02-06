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

    def __init__(self,
                 method=tf.image.ResizeMethod.BILINEAR,
                 antialias=True,
                 mode=None,
                 gamma_kernel_initializer='zeros',
                 gamma_bias_initializer='ones',
                 beta_kernel_initializer='zeros',
                 beta_bias_initializer='zeros',
                 **kwargs):

        """
        @param method: Method used for scaling conditioning in case rank(input) == rank(modulation)
        @type method: tensorflow.image.ResizeMethod
        @param antialias: Use antialiasing if downscaling conditioning
        @type antialias:
        @param mode: one of 'provided_mean_var', 'mapped' or 'provided_meanvar_fused',
                     leave blank to infer on build
        @type mode: str
        """
        kwargs['gamma_kernel_initializer'] = gamma_kernel_initializer
        kwargs['gamma_bias_initializer'] = gamma_bias_initializer
        kwargs['beta_initializer'] = beta_kernel_initializer
        kwargs['beta_bias_initializer'] = beta_bias_initializer
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
                self.add_weight('gamma_kernel', shape=[input_shape[1][-1], input_shape[0][-1]])
                self.add_weight('gamma_bias', shape=[input_shape[0][-1]])
                self.add_weight('beta_kernel', shape=[input_shape[1][-1], input_shape[0][-1]])
                self.add_weight('beta_bias', shape=[input_shape[0][-1]])
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
            x, beta, gamma = x
            beta_gamma = [beta, gamma]
            beta_gamma = tf.concat([beta_gamma], axis=-1)
        elif self.mode == 'mapped':
            x, beta_gamma = x
            beta = tf.matmul(beta_gamma, self.get_weight('beta_kernel', training=training))
            beta = tf.nn.bias_add(beta, self.get_weight('beta_bias', training=training))
            gamma = tf.matmul(beta_gamma, self.get_weight('gamma_kernel', training=training))
            gamma = tf.nn.bias_add(gamma, self.get_weight('gamma_bias', training=training))
            beta_gamma = tf.concat([beta, gamma], axis=-1)
        elif self.mode != 'provided_meanvar_fused':
            raise ValueError('Something is wrong')
        else:
            x, beta_gamma = x
        beta_gamma_shape = shape_list(beta_gamma)
        x_shape = shape_list(x)
        if len(beta_gamma_shape) != len(x_shape):
            beta_gamma = tf.reshape(beta_gamma, [-1, ] + ([1] * (len(x.shape) - 2)) + [2, x_shape[-1]])
        else:
            beta_gamma_shape_npa = np.array(beta_gamma_shape[1:-1])
            x_shape_npa = np.array(x_shape[1:-1])
            compatible = np.all(np.logical_or(beta_gamma_shape_npa == 1, beta_gamma_shape_npa == x_shape_npa))
            if not compatible:
                size = np.where(beta_gamma_shape_npa == 1, beta_gamma_shape_npa, x_shape_npa).tolist()
                if len(beta_gamma_shape) == 4:
                    beta_gamma = tf.image.resize(beta_gamma, size, method=self.method)
                elif len(beta_gamma_shape) == 3:
                    beta_gamma = tf.squeeze(
                        tf.image.resize(tf.expand_dims(beta_gamma, 1), [1] + size, method=self.method,
                                        antialias=self.antialias), axis=1)
                else:
                    raise ValueError('Only works for 1D or 2D tensors')

            shape = [beta_gamma_shape[0]] + np.where(beta_gamma_shape_npa == 1, beta_gamma_shape_npa,
                                                     x_shape_npa).tolist() + [2, x_shape[-1]]
            beta_gamma = tf.reshape(beta_gamma, shape)
        beta, gamma = tf.unstack(beta_gamma, axis=-2)

        return (x * gamma) + beta

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

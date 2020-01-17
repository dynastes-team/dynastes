import numpy as np
import six
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object, deserialize_keras_object

from dynastes.ops.t2t_common import shape_list
from dynastes.weight_normalizers.spectral import SpectralNormalization


def serialize(normalizer):
    return serialize_keras_object(normalizer)


def deserialize(config, custom_objects={}):
    custom_objects = {**custom_objects, **object_scope}
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='normalizer')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        identifier = str(identifier)
        # We have to special-case functions that return classes.
        # TODO(omalleyt): Turn these into classes or class aliases.
        if identifier == 'spectral':
            return SpectralNormalization()
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret normalizer identifier:', identifier)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class WscaleNormalizer(tfkl.Layer):

    def __init__(self,
                 lrmul=1.,
                 gain=np.sqrt(2),
                 next_layer=tfkl.Activation('linear'),
                 trainable=False,
                 **kwargs):
        if 'trainable' in kwargs:
            kwargs.pop('trainable')
        super(WscaleNormalizer, self).__init__(trainable=False, **kwargs)
        self.next_layer = get(next_layer)
        self.gain = gain
        self.lrmul = lrmul

    def call(self, inputs, training=None):
        input_shape = shape_list(inputs)
        if len(input_shape) > 1:
            fan_in = np.prod(input_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
            he_std = self.gain / np.sqrt(fan_in)  # He init
            runtime_coef = he_std * self.lrmul
        else:
            runtime_coef = self.lrmul
        return self.next_layer(inputs * runtime_coef, training=training)

    def get_config(self):
        config = {
            'lrmul': self.lrmul,
            'gain': self.gain,
            'next_layer': serialize(self.next_layer)
        }
        base_config = super(WscaleNormalizer, self).get_config()
        return {**base_config, **config}


object_scope = {
    'SpectralNormalization': SpectralNormalization,
    'WscaleNormalizer': WscaleNormalizer,
    'Activation': tfkl.Activation
}

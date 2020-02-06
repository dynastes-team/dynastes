import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object, deserialize_keras_object
from dynastes.ops.t2t_common import shape_list

@tf.keras.utils.register_keras_serializable(package='Dynastes', name='GLU')
def GLU(x, g, y=None):
    _g = tf.sigmoid(g)
    _y = x * _g
    if y is not None:
        _y += (1 - _g) * y
    return _y


# GLUM is to GLU as Mish is to Swish
@tf.keras.utils.register_keras_serializable(package='Dynastes', name='GLUM')
def GLUM(x, g, y=None):
    _g = tf.math.tanh(tf.math.softplus(g))
    _y = x * _g
    if y is not None:
        _y += (1 - _g) * y
    return _y


@tf.keras.utils.register_keras_serializable(package='Dynastes', name='GDOT')
def GDOT(x, g, y=None):
    _g = tf.matmul(x, g)
    _y = x * _g
    if y is not None:
        _y += (1 - _g) * y
    return _y


def _get(gating_function):
    if type(gating_function) == str:
        if gating_function in ['Dynastes>GLUM', 'GLUM']:
            return GLUM
        elif gating_function in ['Dynastes>GLU', 'GLU']:
            return GLUM
        elif gating_function in ['Dynastes>GDOT', 'GDOT']:
            return GDOT
    return _deserialize(gating_function)


def _serialize(gating_function):
    return serialize_keras_object(gating_function)


def _deserialize(config, custom_objects={}):
    custom_objects = {**custom_objects, **object_scope}
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='gating_functions')


object_scope = {
    'GLU': GLU,
    'GLUM': GLUM,
}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class GatingLayer(tfkl.Layer):

    def __init__(self,
                 gating_function='GLU',
                 **kwargs):
        super(GatingLayer, self).__init__(**kwargs)
        self.gating_function = _get(gating_function)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if type(mask) == list:
                return mask[0]
        return mask

    def call(self, inputs, training=None, **kwargs):
        y = None
        if type(inputs) == list:
            x, g = inputs
            d_x = shape_list(x)[-1]
            d_g = shape_list(g)[-1]
            if d_g // 2 == d_x:
                y, g = tf.split(g, num_or_size_splits=2, axis=-1)
            else:
                assert d_g == d_x
        else:
            x, g = tf.split(inputs, num_or_size_splits=2, axis=-1)
        return self.gating_function(x, g, y=y)

    def compute_output_shape(self, input_shape):
        if type(input_shape) == list:
            return input_shape[0]
        output_shape = input_shape
        output_shape[-1] = output_shape[-1] // 2
        return output_shape

    def get_config(self):
        config = {
            'gating_function': _serialize(self.gating_function),
        }
        base_config = super(GatingLayer, self).get_config()
        return {**base_config, **config}


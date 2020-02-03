from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import activations as tf_activations
from tensorflow.python.keras.activations import serialize as _serialize

tfaa = None
try:
    import tensorflow_addons.activations as tfaa
except:
    print('WARNING! TensorFlow Addons are missing!')
    tfal = None


@tf.keras.utils.register_keras_serializable(package='Dynastes', name='swish')
def swish(x):
    return tf.nn.swish(x)


mish = None
rrelu = None

if tfaa is not None:
    mish = tfaa.mish
    rrelu = tfaa.rrelu


def get(activation):
    if type(activation) == str:
        if activation in ['Dynastes>swish', 'swish']:
            return swish
        elif tfaa is not None:
            if activation in ['Addons>mish', 'mish']:
                return mish
            elif activation in ['Addons>rrelu', 'rrelu']:
                return rrelu
    return tf_activations.get(activation)


def serialize(activation):
    return _serialize(activation)

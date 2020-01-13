from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.keras import activations as tf_activations
from tensorflow.python.keras.activations import deserialize as _deserialize
from tensorflow.python.keras.activations import serialize as _serialize
tfaa = None
try:
    import tensorflow_addons.activations as tfaa
except:
    print('WARNING! TensorFlow Addons are missing!')
    tfal = None

def swish(x, beta=1):
    return x * tf.math.sigmoid(beta * x)


object_scope = {
    'swish': swish
}

mish = None
rrelu = None

if tfaa is not None:
    mish = tfaa.mish
    rrelu = tfaa.rrelu

    object_scope['mish'] = tfaa.mish
    object_scope['rrelu'] = tfaa.rrelu

def serialize(activation):
    if activation == swish:
        return 'swish'
    elif tfaa is not None:
        if activation == mish:
            return 'mish'
    return _serialize(activation)


def deserialize(name, custom_objects=None):
    if custom_objects is None:
        custom_objects = {}
    custom_objects = {**custom_objects, **object_scope}
    return _deserialize(name, custom_objects)


def get(activation):
    if type(activation) == str:
        if activation == 'swish':
            return swish
        elif tfaa is not None:
            if activation == 'mish':
                return mish
    with custom_object_scope(object_scope):
        return tf_activations.get(activation)

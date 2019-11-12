from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import regularizers as tf_regularizers
from tensorflow.python.keras.regularizers import deserialize as _deserialize
from tensorflow.python.keras.regularizers import serialize as _serialize

from . import orthogonal
from .orthogonal import Orthogonal


def serialize(regularizer):
    return _serialize(regularizer)


def deserialize(config, custom_objects=None):
    custom_objects = {**custom_objects, **{'Orthogonal': Orthogonal}}
    return _deserialize(config, custom_objects)


def get(regularizer):
    if type(regularizer) == str:
        if regularizer == 'orthogonal':
            return Orthogonal()
    return tf_regularizers.get(regularizer)


# Cleanup symbols to avoid polluting namespace.
del absolute_import
del division
del print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.keras import regularizers as tf_regularizers
from tensorflow.python.keras.regularizers import deserialize as _deserialize
from tensorflow.python.keras.regularizers import serialize as _serialize
from tensorflow.keras.regularizers import Regularizer

from . import orthogonal
from .orthogonal import Orthogonal

object_scope = {
    'Orthogonal': Orthogonal
}


class ModifyingRegularizer(Regularizer):

    def __init__(self, first, second,
                 fn,
                 **kwargs):
        super(ModifyingRegularizer).__init__(**kwargs)
        self.first = get(first)
        self.second = get(second)
        self.fn = fn

    def __call__(self, x):
        return self.fn(self.first(x), self.second(x))

    def get_config(self):
        return {'first': serialize(self.first),
                'second': serialize(self.second)}


def _add(self, other):
    return ModifyingRegularizer(self, other, lambda x, y: x + y)


def _sub(self, other):
    return ModifyingRegularizer(self, other, lambda x, y: x - y)


setattr(Regularizer, '__add__', _add)
setattr(Regularizer, '__sub__', _sub)


def serialize(regularizer):
    return _serialize(regularizer)


def deserialize(config, custom_objects={}):
    custom_objects = {**custom_objects, **object_scope}
    return _deserialize(config, custom_objects)


def get(regularizer):
    if type(regularizer) == str:
        if regularizer == 'orthogonal':
            return Orthogonal()
    with custom_object_scope(object_scope):
        return tf_regularizers.get(regularizer)


# Cleanup symbols to avoid polluting namespace.
del absolute_import
del division
del print_function

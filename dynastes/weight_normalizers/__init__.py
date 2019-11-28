from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object

from .spectral import SpectralNormalization

object_scope = {
    'SpectralNormalization': SpectralNormalization,
}


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
        raise ValueError('Could not interpret regularizer identifier:', identifier)


def serialize(regularizer):
    return serialize_keras_object(regularizer)


def deserialize(config, custom_objects={}):
    custom_objects = {**custom_objects, **object_scope}
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='regularizer')


# Cleanup symbols to avoid polluting namespace.
del absolute_import
del division
del print_function

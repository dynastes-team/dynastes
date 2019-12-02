from abc import ABCMeta
from functools import partial
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfpb = tfp.bijectors


def _get_roll_op(axis=0, steps=1, constant=0):
    def roll_op(x):
        return tf.stack([tf.roll(_x, -((i * steps) + constant), axis) for i, _x in enumerate(tf.unstack(x, axis=axis))],
                        axis=axis)

    return roll_op


class RollIncremental(tfpb.Bijector, metaclass=ABCMeta):

    def __init__(
            self,
            axis,
            steps=0,
            constant=0,
            **kwargs):
        kwargs['forward_min_event_ndims'] = axis + 1
        kwargs['inverse_min_event_ndims'] = axis + 1
        super(RollIncremental, self).__init__(**kwargs)
        setattr(self, '_forward', _get_roll_op(axis=axis, steps=steps, constant=constant))
        setattr(self, '_inverse', _get_roll_op(axis=axis, steps=-steps, constant=-constant))
        setattr(self, 'forward_event_shape', lambda x: x)
        setattr(self, 'inverse_event_shape', lambda x: x)


class EventShapeAwareChain(tfpb.Chain):

    def __init__(self,
                 event_shape_in,
                 partial_bijectors: List[partial],
                 **kwargs):

        bijectors = []
        for partial_bijector in partial_bijectors:
            if partial_bijector.func == tfpb.Reshape:
                if partial_bijector.keywords['event_shape_out'][-1] == -1:
                    if np.cumprod(np.array(partial_bijector.keywords['event_shape_out'][:-1]))[-1] == \
                            np.cumprod(np.array(event_shape_in[:-1]))[-1]:
                        partial_bijector.keywords['event_shape_out'][-1] = event_shape_in[-1]
                bijector = partial_bijector(event_shape_in=event_shape_in)
            else:
                bijector = partial_bijector()
            event_shape_in = bijector.forward_event_shape(event_shape_in)
            bijectors.append(bijector)
        kwargs['bijectors'] = list(reversed(bijectors))
        super(EventShapeAwareChain, self).__init__(**kwargs)

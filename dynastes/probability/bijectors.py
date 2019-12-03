import copy
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


def _fix_unknown_dimension(input_shape, output_shape):
    """Find and replace a missing dimension in an output shape.
    This is a near direct port of the internal Numpy function
    `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`
    Arguments:
      input_shape: Shape of array being reshaped
      output_shape: Desired shape of the array with at most
        a single -1 which indicates a dimension that should be
        derived from the input shape.
    Returns:
      The new output shape with a -1 replaced with its computed value.
    Raises:
      ValueError: If the total array size of the output_shape is
      different than the input_shape, or more than one unknown dimension
      is specified.
    """
    output_shape = list(output_shape)
    msg = 'total size of new array must be unchanged'

    known, unknown = 1, None
    for index, dim in enumerate(output_shape):
        if dim < 0:
            if unknown is None:
                unknown = index
            else:
                raise ValueError('Can only specify one unknown dimension.')
        else:
            known *= dim

    original = np.prod(input_shape, dtype=int)
    if unknown is not None:
        if known == 0 or original % known != 0:
            raise ValueError(msg)
        output_shape[unknown] = original // known
    elif original != known:
        raise ValueError(msg)
    return output_shape


def _process_event_shapes(event_shape_in, event_shape_out, ch):
    event_shape_out = np.array(event_shape_out)
    event_shape_in = np.array(event_shape_in)
    infer_indices = np.where(event_shape_out == -1)[0]
    if len(infer_indices) == 0:
        return event_shape_out.tolist()
    elif len(infer_indices) == 1:
        if event_shape_out[-1] == -1:
            event_shape_out[-1] = ch
            return event_shape_out
        return _fix_unknown_dimension(event_shape_in.tolist(), event_shape_out.tolist())
    else:
        if event_shape_out[-1] == -1:
            event_shape_out[-1] = ch
        return _process_event_shapes(event_shape_in.tolist(), event_shape_out.tolist(), ch)


class EventShapeAwareChain(tfpb.Chain):

    def __init__(self,
                 event_shape_in,
                 partial_bijectors: List[partial],
                 **kwargs):

        bijectors = []
        ch = event_shape_in[-1]
        for partial_bijector in partial_bijectors:
            if partial_bijector.func == tfpb.Reshape:
                partial_bijector = partial(partial_bijector.func,
                                           **{k: copy.copy(v) for k, v in partial_bijector.keywords.items()})
                partial_bijector.keywords['event_shape_out'] = _process_event_shapes(event_shape_in,
                                                                                     partial_bijector.keywords[
                                                                                         'event_shape_out'], ch)
                bijector = partial_bijector(event_shape_in=event_shape_in)
            else:
                bijector = partial_bijector()
            event_shape_in = bijector.forward_event_shape(event_shape_in)
            bijectors.append(bijector)
        kwargs['bijectors'] = list(reversed(bijectors))
        super(EventShapeAwareChain, self).__init__(**kwargs)

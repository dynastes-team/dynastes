from functools import partial

import tensorflow_probability as tfp

tfpb = tfp.bijectors
from dynastes.probability import bijectors


def Reshape(event_shape_out) -> partial:
    return partial(tfpb.Reshape, event_shape_out=event_shape_out)


def RollIncremental(axis,
                    steps=0,
                    constant=0) -> partial:
    return partial(bijectors.RollIncremental, axis=axis, steps=steps, constant=constant)


def Transpose(perm) -> partial:
    return partial(tfpb.Transpose, perm=perm)

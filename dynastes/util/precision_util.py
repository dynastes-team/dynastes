from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def large_compatible_negative(tensor_type):
  """Large negative number as Tensor.
  This function is necessary because the standard value for epsilon
  in this module (-1e9) cannot be represented using tf.float16
  Args:
    tensor_type: a dtype to determine the type.
  Returns:
    a large negative number.
  """
  if tensor_type == tf.float16:
    return tf.float16.min
  return -1e9
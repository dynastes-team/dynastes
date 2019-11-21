import tensorflow as tf
import numpy as np

def to_float(x):
  """Cast x to float; created because tf.to_float is deprecated."""
  return tf.cast(x, tf.float32)

def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    x_name = "(eager Tensor)"
    try:
      x_name = x.name
    except AttributeError:
      pass
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                       x.device, cast_x.device)
  return cast_x

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def dropout_with_broadcast_dims(x, rate, broadcast_dims=None, **kwargs):
  """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.
  Instead of specifying noise_shape, this function takes broadcast_dims -
  a list of dimension numbers in which noise_shape should be 1.  The random
  keep/drop tensor has dimensionality 1 along these dimensions.
  Args:
    x: a floating point tensor.
    keep_prob: A scalar Tensor with the same type as x.
      The probability that each element is kept.
    broadcast_dims: an optional list of integers
      the dimensions along which to broadcast the keep/drop flags.
    **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".
  Returns:
    Tensor of the same shape as x.
  """
  assert "noise_shape" not in kwargs
  if broadcast_dims:
    shape = tf.shape(x)
    ndims = len(x.get_shape())
    # Allow dimensions like "-1" as well.
    broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
    kwargs["noise_shape"] = [
        1 if i in broadcast_dims else shape[i] for i in range(ndims)
    ]
  return tf.nn.dropout(x, rate, **kwargs)

def top_kth_iterative(x, k):
  """Compute the k-th top element of x on the last axis iteratively.
  This assumes values in x are non-negative, rescale if needed.
  It is often faster than tf.nn.top_k for small k, especially if k < 30.
  Note: this does not support back-propagation, it stops gradients!
  Args:
    x: a Tensor of non-negative numbers of type float.
    k: a python integer.
  Returns:
    a float tensor of the same shape as x but with 1 on the last axis
    that contains the k-th largest number in x.
  """
  # The iterative computation is as follows:
  #
  # cur_x = x
  # for _ in range(k):
  #   top_x = maximum of elements of cur_x on the last axis
  #   cur_x = cur_x where cur_x < top_x and 0 everywhere else (top elements)
  #
  # We encode this computation in a TF graph using tf.foldl, so the inner
  # part of the above loop is called "next_x" and tf.foldl does the loop.
  def next_x(cur_x, _):
    top_x = tf.reduce_max(cur_x, axis=-1, keep_dims=True)
    return cur_x * to_float(cur_x < top_x)
  # We only do k-1 steps of the loop and compute the final max separately.
  fin_x = tf.foldl(next_x, tf.range(k - 1), initializer=tf.stop_gradient(x),
                   parallel_iterations=2, back_prop=False)
  return tf.stop_gradient(tf.reduce_max(fin_x, axis=-1, keep_dims=True))

def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """Matrix band part of ones.
  Args:
    rows: int determining number of rows in output
    cols: int
    num_lower: int, maximum distance backward. Negative values indicate
      unlimited.
    num_upper: int, maximum distance forward. Negative values indicate
      unlimited.
    out_shape: shape to reshape output by.
  Returns:
    Tensor of size rows * cols reshaped into shape out_shape.
  """
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = tf.constant(band, tf.float32)
  else:
    band = tf.linalg.band_part(
        tf.ones([rows, cols]), tf.cast(num_lower, tf.int64),
        tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band
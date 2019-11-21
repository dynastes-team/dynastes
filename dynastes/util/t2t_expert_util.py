import tensorflow as tf

def to_float(x):
  """Cast x to float; created because to_float is deprecated."""
  return tf.cast(x, tf.float32)

def to_int32(x):
  """Cast x to float; created because to_float is deprecated."""
  return tf.cast(x, tf.int32)


class TruncatingDispatcher(object):
  """Helper for implementing a mixture of experts.
  A TruncatingDispatcher is useful when you need to deal with
  fixed-sized Tensors.  As opposed to a SparseDispatcher, which
  produces batches of different sizes for the different experts, the
  TruncatingDispatcher always produces batches of the same given size,
  and the results are returned stacked in one big tensor.
  In the case where an expert is over-capacity, the last items that
  should have gone to that expert are dropped.
  Confusingly, the inputs to a TruncatingDispatcher have both a
  "batch" and a "length" dimension.  Not only does each expert receive
  the same total number of examples, it also receives the same number
  of examples for each element of "batch".  This behavior is necessary
  for applications such as grouped attention, where we have a batch of
  sequences, and we want each sequence to be divided evenly among
  experts.  For simpler applications like mixture-of-experts, you can
  reshape the input so that the "batch" dimension is 1, and only the
  "length" dimension is used.
  """

  def __init__(self, requests, expert_capacity):
    """Create a TruncatingDispatcher.
    Args:
      requests: a boolean `Tensor` of shape `[batch, length, num_experts]`.
        Alternatively, a float or int Tensor containing zeros and ones.
      expert_capacity: a Scalar - maximum number of examples per expert per
        batch element.
    Returns:
      a TruncatingDispatcher
    """
    self._requests = to_float(requests)
    self._expert_capacity = expert_capacity
    expert_capacity_f = to_float(expert_capacity)
    self._batch, self._length, self._num_experts = tf.unstack(
        tf.shape(self._requests), num=3)

    # [batch, length, num_experts]
    position_in_expert = tf.cumsum(self._requests, axis=1, exclusive=True)
    # [batch, length, num_experts]
    self._gates = self._requests * to_float(
        tf.less(position_in_expert, expert_capacity_f))
    batch_index = tf.reshape(
        to_float(tf.range(self._batch)), [self._batch, 1, 1])
    length_index = tf.reshape(
        to_float(tf.range(self._length)), [1, self._length, 1])
    expert_index = tf.reshape(
        to_float(tf.range(self._num_experts)), [1, 1, self._num_experts])
    # position in a Tensor with shape [batch * num_experts * expert_capacity]
    flat_position = (
        position_in_expert +
        batch_index * (to_float(self._num_experts) * expert_capacity_f) +
        expert_index * expert_capacity_f)
    # Tensor of shape [batch * num_experts * expert_capacity].
    # each element is an integer in [0, length)
    self._indices = tf.math.unsorted_segment_sum(
        data=tf.reshape((length_index + 1.0) * self._gates, [-1]),
        segment_ids=to_int32(tf.reshape(flat_position, [-1])),
        num_segments=self._batch * self._num_experts * expert_capacity)
    self._indices = tf.reshape(
        self._indices,
        [self._batch, self._num_experts, expert_capacity])
    # Tensors of shape [batch, num_experts, expert_capacity].
    # each element is 0.0 or 1.0
    self._nonpadding = tf.minimum(self._indices, 1.0)
    # each element is an integer in [0, length)
    self._indices = tf.nn.relu(self._indices - 1.0)
    # self._flat_indices is [batch, num_experts, expert_capacity], with values
    # in [0, batch * length)
    self._flat_indices = to_int32(
        self._indices +
        (tf.reshape(to_float(tf.range(self._batch)), [-1, 1, 1])
         * to_float(self._length)))
    self._indices = to_int32(self._indices)

  def dispatch(self, inp):
    """Send the inputs to the experts.
    Args:
      inp: a `Tensor` of shape "[batch, length, depth]`
    Returns:
      a tensor with shape [batch, num_experts, expert_capacity, depth]
    """
    inp = tf.reshape(inp, [self._batch * self._length, -1])
    # [batch, num_experts, expert_capacity, depth]
    ret = tf.gather(inp, self._flat_indices)
    return ret

  def combine(self, x):
    """Return the output from the experts.
    When one example goes to multiple experts, the outputs are summed.
    Args:
      x: a Tensor with shape [batch, num_experts, expert_capacity, depth]
    Returns:
      a `Tensor` with shape `[batch, length, depth]
    """
    depth = tf.shape(x)[-1]
    x *= tf.expand_dims(self._nonpadding, -1)
    ret = tf.math.unsorted_segment_sum(
        x, self._flat_indices, num_segments=self._batch * self._length)
    ret = tf.reshape(ret, [self._batch, self._length, depth])
    return ret

  def nonpadding(self):
    """Which elements of a dispatched Tensor are not padding.
    Returns:
      a Zero/One float tensor with shape [batch, num_experts, expert_capacity].
    """
    return self._nonpadding

  def gates(self):
    """A Tensor indicating which examples go to which experts.
    Returns:
      A float32 Tensor with shape [batch, length, num_experts], where each value
      is 0.0 or 1.0.
    """
    return self._gates

  def length_coordinate(self):
    """Length coordinate of dispatched tensor.
    Returns:
      a tensor with shape [batch, num_experts, expert_capacity] containing
       integers in the range [0, length)
    """
    return self._indices
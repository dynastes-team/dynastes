import tensorflow as tf
from tensorflow.python.framework import function

from dynastes.ops.t2t_common import cast_like


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """Identity operation whose gradient is converted to a `Tensor`.
    Currently, the gradient to `tf.concat` is particularly expensive to
    compute if dy is an `IndexedSlices` (a lack of GPU implementation
    forces the gradient operation onto CPU).  This situation occurs when
    the output of the `tf.concat` is eventually passed to `tf.gather`.
    It is sometimes faster to convert the gradient to a `Tensor`, so as
    to get the cheaper gradient for `tf.concat`.  To do this, replace
    `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.
    Args:
      x: A `Tensor`.
    Returns:
      The input `Tensor`.
    """
    return x


def dropout_no_scaling(x, keep_prob):
    """Like tf.nn.dropout, but does not scale up.  Works on integers also.
    Args:
      x: a Tensor
      keep_prob: a floating point number
    Returns:
      Tensor of the same shape as x.
    """
    if keep_prob == 1.0:
        return x
    mask = tf.less(tf.random_uniform(tf.shape(x)), keep_prob)
    return x * cast_like(mask, x)


def reshape_like(a, b):
    """Reshapes a to match the shape of b in all but the last dimension."""
    ret = tf.reshape(a, tf.concat([tf.shape(b)[:-1], tf.shape(a)[-1:]], 0))
    if not tf.executing_eagerly():
        ret.set_shape(b.get_shape().as_list()[:-1] + a.get_shape().as_list()[-1:])
    return ret


def gather(params, indices, dtype=tf.float32):
    """Version of tf.gather that works faster on tpu."""
    if not tf.config.optimizer.get_jit():
        return tf.gather(params, indices)
    vocab_size = params.get_shape().as_list()[0]
    indices_flat = tf.reshape(indices, [-1])
    out = tf.matmul(tf.one_hot(indices_flat, vocab_size, dtype=dtype), params)
    out = reshape_like(out, tf.expand_dims(indices, -1))
    return out


def embedding_lookup(x,
                     embedding_matrix=None,
                     name='embedding_lookup',
                     multiplier=1.0,
                     symbol_dropout_rate=0.0,
                     dtype=tf.float32):
    """Embed x of type int64 into dense vectors, reducing to max 4 dimensions."""
    with tf.name_scope(name):
        # On the backwards pass, we want to convert the gradient from
        # an indexed-slices to a regular tensor before sending it back to the
        # parameter server. This avoids excess computation on the parameter server.
        if not tf.executing_eagerly():
            embedding_matrix = convert_gradient_to_tensor(embedding_matrix)
        x = dropout_no_scaling(x, 1.0 - symbol_dropout_rate)
        emb_x = gather(embedding_matrix, x, dtype)
        if multiplier != 1.0:
            emb_x *= multiplier
        static_shape = emb_x.shape.as_list()
        if len(static_shape) < 5:
            return emb_x
        assert len(static_shape) == 5
        # If we had an extra channel dimension, assume it's 1, i.e. shape[3] == 1.
        return tf.squeeze(emb_x, 3)

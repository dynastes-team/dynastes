from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights

    Source:
    https://www.tensorflow.org/tutorials/text/transformer
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def split_heads(x, num_heads, ignore_dims=0):
    """Split the last dimension into (num_heads, depth).

    Source (modified):
    https://www.tensorflow.org/tutorials/text/transformer
    """
    shape = x.shape.as_list()
    x = tf.reshape(x, tuple(shape[:-1]) + (num_heads, shape[-1] // num_heads))
    dims = list(range(len(shape) + 1))
    perm = dims[:1 + ignore_dims] + [dims[-2]] + dims[1 + ignore_dims:-2] + [dims[-1]]
    return tf.transpose(x, perm=perm)


def merge_heads(x, ignore_dims=0):
    """
    Source (modified):
    https://www.tensorflow.org/tutorials/text/transformer
    """

    shape = x.shape.as_list()
    dims = list(range(len(shape)))
    perm = dims[:1 + ignore_dims] + [dims[-2]] + [dims[-3]] + [dims[-1]]
    x = tf.transpose(x, perm=perm)
    return tf.reshape(x, shape[:1 + ignore_dims] + [shape[-2]] + [shape[-3] * shape[-1]])

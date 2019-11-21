from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dynastes.util.precision_util import large_compatible_negative


def scaled_dot_product_attention(q, k, v, bias, multiquery_attention=False):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      bias: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
      multiquery_attention: Use one head for K and V,
            see https://arxiv.org/abs/1911.02150v1

    Returns:
      output, attention_weights

    Source:
    https://www.tensorflow.org/tutorials/text/transformer
    """
    leading_dims = ['b', 'x', 'y', 'z', 'q', 'a', 'c']
    shape = q.shape.as_list()
    ldim = ''.join(leading_dims[:len(shape) - 3])
    ld3 = (ldim, ldim, ldim)

    if multiquery_attention:
        logits = tf.einsum("%shnk,%smk->%shnm" % ld3, q, k)
    else:
        logits = tf.einsum("%shnk,%shmk->%shnm" % ld3, q, k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = logits / tf.math.sqrt(dk)

    # add the bias to the scaled tensor.
    if bias is not None:
        scaled_attention_logits += bias

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    if multiquery_attention:
        output = tf.einsum("%shnm,%smv->%shnv" % ld3, attention_weights, v)
    else:
        output = tf.einsum("%shnm,%shmv->%shnv" % ld3, attention_weights, v)

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

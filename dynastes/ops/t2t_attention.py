import math

import tensorflow as tf

from dynastes.ops import t2t_common
from dynastes.util import t2t_expert_util


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
      x: a Tensor with shape [..., m]
      n: an integer.
    Returns:
      a Tensor with shape [..., n, m/n]
    """
    x_shape = t2t_common.shape_list(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
      x: a Tensor with shape [..., a, b]
    Returns:
      a Tensor with shape [..., ab]
    """
    x_shape = t2t_common.shape_list(x)
    a, b = x_shape[-2:]
    return tf.reshape(x, x_shape[:-2] + [a * b])


def combine_first_two_dimensions(x):
    """Reshape x so that the first two dimension become one.
    Args:
      x: a Tensor with shape [a, b, ...]
    Returns:
      a Tensor with shape [ab, ...]
    """
    ret = tf.reshape(x, tf.concat([[-1], t2t_common.shape_list(x)[2:]], 0))
    old_shape = x.get_shape().dims
    a, b = old_shape[:2]
    new_shape = [a * b if a and b else None] + old_shape[2:]
    ret.set_shape(new_shape)
    return ret


def split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).
    Args:
      x: a Tensor with shape [batch, length, channels]
      num_heads: an integer
    Returns:
      a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def split_heads_2d(x, num_heads):
    """Split channels (dimension 3) into multiple heads (becomes dimension 1).
    Args:
      x: a Tensor with shape [batch, height, width, channels]
      num_heads: an integer
    Returns:
      a Tensor with shape [batch, num_heads, height, width, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 3, 1, 2, 4])


def split_heads_nd(x, num_heads):
    """Split the depth dimension (last dimension) into multiple heads.
    Args:
      x: a [batch, d1, ..., dn, depth] tensor
      num_heads: an integer
    Returns:
      a [batch, num_heads, d1, ..., dn, depth // num_heads]
    """
    num_dimensions = len(t2t_common.shape_list(x)) - 2
    return tf.transpose(
        split_last_dimension(x, num_heads), [0, num_dimensions + 1] +
                                            list(range(1, num_dimensions + 1)) + [num_dimensions + 2])


def combine_heads(x):
    """Inverse of split_heads.
    Args:
      x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
    Returns:
      a Tensor with shape [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def combine_heads_2d(x):
    """Inverse of split_heads_2d.
    Args:
      x: a Tensor with shape
        [batch, num_heads, height, width, channels / num_heads]
    Returns:
      a Tensor with shape [batch, height, width, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 1, 4]))


def combine_heads_nd(x):
    """Inverse of split_heads_nd.
    Args:
      x: a [batch, num_heads, d1, ..., dn, depth // num_heads] tensor
    Returns:
      a [batch, d1, ...., dn, depth] tensor
    """
    num_dimensions = len(t2t_common.shape_list(x)) - 3
    return combine_last_two_dimensions(
        tf.transpose(x, [0] + list(range(2, num_dimensions + 2)) +
                     [1, num_dimensions + 2]))


def to_int32(x):
    """Cast x to float; created because to_float is deprecated."""
    return tf.cast(x, tf.int32)


def mixed_precision_is_enabled(
        activation_dtype=None, weight_dtype=None, hparams=None):
    assert not (hparams and (activation_dtype or weight_dtype)), (
        "Provide only hparams or activation_dtype and weight_dtype")
    if (hparams and hasattr(hparams, "activation_dtype") and
            hasattr(hparams, "weight_dtype")):
        activation_dtype = hparams.activation_dtype
        weight_dtype = hparams.weight_dtype
    return activation_dtype == tf.float16 and weight_dtype == tf.float32


def maybe_upcast(logits,
                 activation_dtype=None, weight_dtype=None, hparams=None):
    if mixed_precision_is_enabled(activation_dtype, weight_dtype, hparams):
        return tf.cast(logits, tf.float32)
    return logits


def harden_attention_weights(weights, k, gumbel_noise_weight):
    """Make attention weights non-0 only on the top k ones."""
    if gumbel_noise_weight > 0.:
        gumbel_noise = -tf.log(-tf.log(tf.random_uniform(tf.shape(weights),
                                                         minval=1e-5,
                                                         maxval=1 - 1e-5)))
        weights += gumbel_noise * gumbel_noise_weight

    # Subtract the top-kth weight and zero-out all lower ones.
    # Note that currently in case of numerical ties it will retain more
    # than k elements. In the future, we may want to avoid this.
    weights -= t2t_common.top_kth_iterative(weights, k)
    weights = tf.nn.relu(weights)
    # Re-normalize the weights.
    weights_sum = tf.reduce_sum(weights, axis=-1, keep_dims=True)
    weights_sum = tf.maximum(weights_sum, 1e-6)  # Avoid division by 0.
    weights /= weights_sum
    return weights


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          name='dot_product_attention',
                          dropout_broadcast_dims=[0, 1],
                          activation_dtype=None,
                          weight_dtype=None,
                          hard_attention_k=0,
                          gumbel_noise_weight=0.0,
                          scaled=False):
    """Dot-product attention.
    Args:
      q: Tensor with shape [..., length_q, depth_k].
      k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
        match with q.
      v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
        match with q.
      bias: bias Tensor (see attention_bias())
      dropout_rate: a float.
      save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      dropout_broadcast_dims: an optional list of integers less than rank of q.
        Specifies in which dimensions to broadcast the dropout decisions.
      activation_dtype: Used to define function activation dtype when using
        mixed precision.
      weight_dtype: The dtype weights are stored in when using mixed precision
      hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
      gumbel_noise_weight: if > 0, apply Gumbel noise with weight
        `gumbel_noise_weight` before picking top-k. This is a no op if
        hard_attention_k <= 0.
    Returns:
      Tensor with shape [..., length_q, depth_v].
    """
    dk = tf.cast(t2t_common.shape_list(k)[-1], k.dtype)
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    if scaled:
        logits /= tf.math.sqrt(dk)
    if bias is not None:
        bias = t2t_common.cast_like(bias, logits)
        logits += bias
    # If logits are fp16, upcast before softmax
    logits = maybe_upcast(logits, activation_dtype, weight_dtype)
    weights = tf.nn.softmax(logits, name="attention_weights")
    if hard_attention_k > 0:
        weights = harden_attention_weights(weights, hard_attention_k,
                                           gumbel_noise_weight)
    weights = t2t_common.cast_like(weights, q)
    # Drop out attention links for each head.
    weights = t2t_common.dropout_with_broadcast_dims(
        weights, rate=dropout_rate, broadcast_dims=dropout_broadcast_dims)
    return tf.matmul(weights, v), {name + '/attention_weights': weights}


def get_embedding_initializer_stddev(depth):
    return depth ** -0.5


def get_relative_embeddings_left_right_shape(max_relative_position, depth,
                                             num_heads,
                                             heads_share_relative_embedding):
    max_relative_position_unmasked = 2 * max_relative_position - 1
    if heads_share_relative_embedding:
        embedding_shape = (max_relative_position_unmasked, depth)
    else:
        embedding_shape = (num_heads, max_relative_position_unmasked, depth)
    return embedding_shape


def get_relative_embeddings_left_shape(max_relative_position, depth, num_heads, heads_share_relative_embedding):
    if heads_share_relative_embedding:
        embedding_shape = (max_relative_position, depth)
    else:
        embedding_shape = (num_heads, max_relative_position, depth)
    return embedding_shape


def get_relative_embeddings_left_right(relative_embeddings,
                                       max_relative_position, length, depth,
                                       heads_share_relative_embedding):
    """Instantiate or retrieve relative embeddings, sliced according to length.
    Use for unmasked case where the relative attention looks both left and right.
    Args:
      max_relative_position: an Integer for the number of entries in the relative
        embedding, which corresponds to the max relative distance that is
        considered.
      length: an Integer, specifies the length of the input sequence for which
        this relative embedding is retrieved for.
      depth: an Integer, specifies the depth for relative embeddings.
      num_heads: an Integer, specifies the number of heads.
      heads_share_relative_embedding: a Boolean specifying if the relative
        embedding is shared across heads.
      name: a string giving the name of the embedding variables.
    Returns:
      a Tensor with shape [length, depth]
    """
    # Pad first before slice to avoid using tf.cond.
    pad_length = tf.maximum(length - max_relative_position, 0)
    slice_start_position = tf.maximum(max_relative_position - length, 0)
    if heads_share_relative_embedding:
        padded_relative_embeddings = tf.pad(
            relative_embeddings,
            [[pad_length, pad_length], [0, 0]])
        used_relative_embeddings = tf.slice(
            padded_relative_embeddings,
            [slice_start_position, 0], [2 * length - 1, -1])
    else:
        padded_relative_embeddings = tf.pad(
            relative_embeddings,
            [[0, 0], [pad_length, pad_length], [0, 0]])
        used_relative_embeddings = tf.slice(
            padded_relative_embeddings,
            [0, slice_start_position, 0], [-1, 2 * length - 1, -1])
    return used_relative_embeddings


def get_relative_embeddings_left(relative_embeddings, max_relative_position, length, heads_share_relative_embedding):
    """Instantiate or retrieve relative embeddings, sliced according to length.
    Use for masked case where the relative attention is only looking left.
    Args:
      max_relative_position: an Integer for the number of entries in the relative
        embedding, which corresponds to the max relative distance that is
        considered.
      length: an Integer, specifies the length of the input sequence for which
        this relative embedding is retrieved for.
      depth: an Integer, specifies the depth for relative embeddings.
      num_heads: an Integer, specifies the number of heads.
      heads_share_relative_embedding: a Boolean specifying if the relative
        embedding is shared across heads.
      name: a string giving the name of the embedding variables.
    Returns:
      a Tensor with shape [length, depth]
    """
    # Pad first before slice to avoid using tf.cond.
    pad_length = tf.maximum(length - max_relative_position, 0)
    start_slice_position = tf.maximum(max_relative_position - length, 0)
    if heads_share_relative_embedding:
        padded_relative_embeddings = tf.pad(
            relative_embeddings,
            [[pad_length, 0], [0, 0]])
        used_relative_embeddings = tf.slice(
            padded_relative_embeddings,
            [start_slice_position, 0], [length, -1])
    else:
        padded_relative_embeddings = tf.pad(
            relative_embeddings,
            [[0, 0], [pad_length, 0], [0, 0]])
        used_relative_embeddings = tf.slice(
            padded_relative_embeddings,
            [0, start_slice_position, 0], [-1, length, -1])
    return used_relative_embeddings


def _absolute_position_to_relative_position_unmasked(x):
    """Helper function for dot_product_unmasked_self_attention_relative_v2.
    Rearrange an attention logits or weights Tensor.
    The dimensions of the input represent:
    [batch, heads, query_position, memory_position]
    The dimensions of the output represent:
    [batch, heads, query_position, memory_position - query_position + length - 1]
    Only works with unmasked_attention.
    Args:
      x: a Tensor with shape [batch, heads, length, length]
    Returns:
      a Tensor with shape [batch, heads, length, 2*length-1]
    """
    batch, heads, length, _ = t2t_common.shape_list(x)
    # padd along column
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, length - 1]])
    x_flat = tf.reshape(x, [batch, heads, length ** 2 + length * (length - 1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = tf.pad(x_flat, [[0, 0], [0, 0], [length, 0]])
    x = tf.reshape(x_flat, [batch, heads, length, 2 * length])
    x = tf.slice(x, [0, 0, 0, 1], [batch, heads, length,
                                   2 * length - 1])
    return x


def _relative_position_to_absolute_position_unmasked(x):
    """Converts tensor from relative to aboslute indexing for local attention.
    Args:
      x: a Tensor of shape [batch (or batch*num_blocks), heads,
                            length, 2 * length - 1]
    Returns:
      A Tensor of shape [batch (or batch*num_blocks), heads, length, length]
    """
    x_shape = t2t_common.shape_list(x)
    batch = x_shape[0]
    heads = x_shape[1]
    length = x_shape[2]
    # Concat columns of pad to shift from relative to absolute indexing.
    col_pad = tf.zeros((batch, heads, length, 1))
    x = tf.concat([x, col_pad], axis=3)

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    flat_x = tf.reshape(x, [batch, heads, length * 2 * length])
    flat_pad = tf.zeros((batch, heads, length - 1))
    flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)

    # Reshape and slice out the padded elements.
    final_x = tf.reshape(flat_x_padded, [batch, heads, length + 1, 2 * length - 1])
    final_x = final_x[:, :, :, length - 1:]
    final_x = final_x[:, :, :length, :]
    return final_x


def _absolute_position_to_relative_position_masked(x):
    """Helper to dot_product_self_attention_relative_v2.
    Rearrange an attention logits or weights Tensor.
    The dimensions of the input represent:
    [batch, heads, query_position, memory_position]
    The dimensions of the output represent:
    [batch, heads, query_position, memory_position - query_position + length - 1]
    Only works with masked_attention.  Undefined behavior for regions of the
    input where memory_position > query_position.
    Args:
      x: a Tensor with shape [batch, heads, length, length]
    Returns:
      a Tensor with shape [batch, heads, length, length]
    """
    batch, heads, length, _ = t2t_common.shape_list(x)
    x = tf.pad(x, [[0, 0], [0, 0], [1, 0], [0, 0]])
    x = tf.reshape(x, [batch, heads, length, length + 1])
    x = tf.slice(x, [0, 0, 0, 1], [batch, heads, length, length])
    return x


def _relative_position_to_absolute_position_masked(x):
    """Helper to dot_product_self_attention_relative_v2.
    Rearrange an attention logits or weights Tensor.
    The dimensions of the input represent:
    [batch, heads, query_position, memory_position - query_position + length - 1]
    The dimensions of the output represent:
    [batch, heads, query_position, memory_position]
    Only works with masked_attention.  Undefined behavior for regions of the
    input where memory_position > query_position.
    Args:
      x: a Tensor with shape [batch, heads, length, length]
    Returns:
      a Tensor with shape [batch, heads, length, length]
    """
    batch, heads, length, _ = t2t_common.shape_list(x)
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
    x = tf.reshape(x, [batch, heads, 1 + length, length])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    return x


def matmul_with_relative_values(x, y, heads_share_relative_embedding):
    if heads_share_relative_embedding:
        ret = tf.einsum("bhlm,md->bhld", x, y)
    else:
        ret = tf.einsum("bhlm,hmd->bhld", x, y)
    return ret


def matmul_with_relative_keys(x, y, heads_share_relative_embedding):
    if heads_share_relative_embedding:
        ret = tf.einsum("bhld,md->bhlm", x, y)
    else:
        ret = tf.einsum("bhld,hmd->bhlm", x, y)
    return ret


def dot_product_unmasked_self_attention_relative_v2(
        q, k, v, bias, key_leftright_embeddings, value_leftright_embeddings=None,
        max_relative_position=None, dropout_rate=0.0, save_weights_to=None,
        name='dot_product_unmasked_self_attention_relative_v2',
        dropout_broadcast_dims=[0, 1], heads_share_relative_embedding=False,
        add_relative_to_values=False,
        scaled=False):
    """Calculate relative position-aware dot-product self-attention.
    The attention calculation is augmented with learned representations for the
    relative position between each element in q and each element in k and v.
    Args:
      q: a Tensor with shape [batch, heads, length, depth].
      k: a Tensor with shape [batch, heads, length, depth].
      v: a Tensor with shape [batch, heads, length, depth].
      bias: bias Tensor.
      max_relative_position: an integer the max relative embedding considered.
        Changing this invalidates checkpoints.
      dropout_rate: a floating point number.
      save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      name: an optional string.
      dropout_broadcast_dims:  an optional list of integers less than 4
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.
      heads_share_relative_embedding: a boolean indicating wheather to share
        relative embeddings between attention heads.
      add_relative_to_values: a boolean for whether to add relative component to
        values.
    Returns:
      A Tensor.
    Raises:
      ValueError: if max_relative_position is not > 0.
    """
    if not max_relative_position:
        raise ValueError("Max relative position (%s) should be > 0 when using "
                         "relative self attention." % (max_relative_position))

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    q.get_shape()[2:].assert_is_compatible_with(k.get_shape()[2:])
    q.get_shape()[2:-1].assert_is_compatible_with(v.get_shape()[2:-1])

    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)

    # Use separate embeddings suitable for keys and values.
    _, q_heads, length, _ = t2t_common.shape_list(q)
    _, k_heads, _, depth_k = t2t_common.shape_list(k)
    mqa = k_heads == 1 and q_heads > k_heads

    key_relative_embeddings = get_relative_embeddings_left_right(
        key_leftright_embeddings,
        max_relative_position, length, depth_k,
        heads_share_relative_embedding)
    unmasked_rel_logits = matmul_with_relative_keys(
        q, key_relative_embeddings, heads_share_relative_embedding)
    unmasked_rel_logits = _relative_position_to_absolute_position_unmasked(
        unmasked_rel_logits)
    logits += unmasked_rel_logits
    dk = tf.cast(t2t_common.shape_list(k)[-1], tf.float32)
    if scaled:
        logits /= tf.math.sqrt(dk)
    if bias is not None:
        logits += bias

    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = t2t_common.dropout_with_broadcast_dims(
        weights, rate=dropout_rate, broadcast_dims=dropout_broadcast_dims)
    # relative_weights.set_shape([None, None, None, max_length])
    ret = tf.matmul(weights, v)
    if add_relative_to_values:
        # Adds the contribution of the weighted relative embeddings to the values.
        # [batch, num_heads, query_length, 2*memory_length-1]
        relative_weights = _absolute_position_to_relative_position_unmasked(
            weights)
        depth_v = t2t_common.shape_list(v)[3]
        value_relative_embeddings = get_relative_embeddings_left_right(
            value_leftright_embeddings,
            max_relative_position, length, depth_v, heads_share_relative_embedding)
        ret += matmul_with_relative_values(
            relative_weights, value_relative_embeddings,
            heads_share_relative_embedding)
    return ret, {name + '_attention_weights': weights}


def dot_product_self_attention_relative_v2(q,
                                           k,
                                           v,
                                           bias,
                                           key_left_embedding,
                                           value_left_embedding=None,
                                           max_relative_position=None,
                                           dropout_rate=0.0,
                                           save_weights_to=None,
                                           name='dot_product_self_attention_relative_v2',
                                           dropout_broadcast_dims=[0, 1],
                                           heads_share_relative_embedding=False,
                                           add_relative_to_values=False,
                                           scaled=False):
    """Calculate relative position-aware dot-product self-attention.
    Only works for masked self-attention (no looking forward).
    The attention calculation is augmented with learned representations for the
    relative position between each element in q and each element in k and v.
    Args:
      q: a Tensor with shape [batch, heads, length, depth].
      k: a Tensor with shape [batch, heads, length, depth].
      v: a Tensor with shape [batch, heads, length, depth].
      bias: bias Tensor.
      max_relative_position: an integer indicating the maximum relative distance
        to look back - changing this invalidates checkpoints
      dropout_rate: a floating point number.
      save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      name: an optional string.
      dropout_broadcast_dims:  an optional list of integers less than 4
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.
      heads_share_relative_embedding: a boolean indicating wheather to share
        relative embeddings between attention heads.
      add_relative_to_values: a boolean for whether to add relative component to
        values.
    Returns:
      A Tensor.
    Raises:
      ValueError: if max_relative_position is not > 0.
    """
    if not max_relative_position:
        raise ValueError("Max relative position (%s) should be > 0 when using "
                         "relative self attention." % (max_relative_position))

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    # (Except v can have different depth.)
    # q.get_shape()[2:].assert_is_compatible_with(k.get_shape()[2:])
    # q.get_shape()[2:-1].assert_is_compatible_with(v.get_shape()[2:-1])

    # Use separate embeddings suitable for keys and values.
    _, q_heads, length, _ = t2t_common.shape_list(q)
    _, k_heads, _, _ = t2t_common.shape_list(k)
    mqa = k_heads == 1 and q_heads > k_heads

    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    key_relative_embeddings = get_relative_embeddings_left(key_left_embedding, max_relative_position, length,
                                                           heads_share_relative_embedding)

    rel_logits = matmul_with_relative_keys(q, key_relative_embeddings,
                                           heads_share_relative_embedding)
    rel_logits = _relative_position_to_absolute_position_masked(rel_logits)
    logits += rel_logits
    dk = tf.cast(t2t_common.shape_list(k)[-1], tf.float32)
    if scaled:
        logits /= tf.math.sqrt(dk)
    if bias is not None:
        logits += bias

    weights = tf.nn.softmax(logits, name="attention_weights")
    # Dropping out the attention links for each of the heads.
    weights = t2t_common.dropout_with_broadcast_dims(
        weights, dropout_rate, broadcast_dims=dropout_broadcast_dims)
    output = tf.matmul(weights, v)
    if add_relative_to_values:
        # [batch, num_heads, query_length, memory_length]
        relative_weights = _absolute_position_to_relative_position_masked(weights)
        depth_v = t2t_common.shape_list(v)[3]
        value_relative_embeddings = get_relative_embeddings_left(value_left_embedding, max_relative_position,
                                                                 length,
                                                                 heads_share_relative_embedding)
        output += matmul_with_relative_values(
            relative_weights, value_relative_embeddings,
            heads_share_relative_embedding)
    return output, {name + '_attention_weights': weights}


def embedding_to_padding(emb):
    """Calculates the padding bias based on which embeddings are all zero.
    We have hacked symbol_modality to return all-zero embeddings for padding.
    Args:
      emb: a Tensor with shape [..., depth].
    Returns:
      a float Tensor with shape [...]. Each element is 1 if its corresponding
      embedding vector is all zero, and is 0 otherwise.
    """
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return t2t_common.to_float(tf.equal(emb_sum, 0.0))


def reshape_by_blocks(x, x_shape, memory_block_size):
    """Reshapes input by splitting its length over blocks of memory_block_size.
    Args:
      x: a Tensor with shape [batch, heads, length, depth]
      x_shape: tf.TensorShape of x.
      memory_block_size: Integer which divides length.
    Returns:
      Tensor with shape
      [batch, heads, length // memory_block_size, memory_block_size, depth].
    """
    x = tf.reshape(x, [
        x_shape[0], x_shape[1], x_shape[2] // memory_block_size,
        memory_block_size, x_shape[3]
    ])
    return x


def attention_bias_local(length, max_backward, max_forward):
    """Create an bias tensor to be added to attention logits.
    A position may attend to positions at most max_distance from it,
    forward and backwards.
    This does not actually save any computation.
    Args:
      length: int
      max_backward: int, maximum distance backward to attend. Negative values
        indicate unlimited.
      max_forward: int, maximum distance forward to attend. Negative values
        indicate unlimited.
    Returns:
      a `Tensor` with shape [1, 1, length, length].
    """
    band = t2t_common.ones_matrix_band_part(
        length,
        length,
        max_backward,
        max_forward,
        out_shape=[1, 1, length, length])
    return -1e9 * (1.0 - band)


def attention_bias_lower_triangle(length):
    """Create an bias tensor to be added to attention logits.
    Allows a query to attend to all positions up to and including its own.
    Args:
     length: a Scalar.
    Returns:
      a `Tensor` with shape [1, 1, length, length].
    """
    return attention_bias_local(length, -1, 0)


def local_attention_1d(q, k, v, block_length=128, filter_width=100, name='local_attention_1d', scaled=False):
    """Strided block local self-attention.
    The sequence is divided into blocks of length block_length. Attention for a
    given query position can see all memory positions in the corresponding block
    and filter_width many positions to the left and right of the block.
    Args:
      q: a Tensor with shape [batch, heads, length, depth_k]
      k: a Tensor with shape [batch, heads, length, depth_k]
      v: a Tensor with shape [batch, heads, length, depth_v]
      block_length: an integer
      filter_width: an integer indicating how much to look left and right of the
        block.
      name: an optional string
    Returns:
      a Tensor of shape [batch, heads, length, depth_v]
    """
    # Check that q, k, v have the same shape except in their depth dimension.
    q.get_shape()[2:-1].assert_is_compatible_with(k.get_shape()[2:-1])
    q.get_shape()[2:-1].assert_is_compatible_with(v.get_shape()[2:-1])

    batch_size, num_heads, original_length, _ = t2t_common.shape_list(q)

    # Pad query, key, value to ensure multiple of corresponding lengths.
    def pad_to_multiple(x, pad_length):
        x_length = t2t_common.shape_list(x)[2]
        return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l_and_r(x, pad_length):
        return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

    # Set up query blocks.
    # [batch, heads, blocks_q, block_length, depth_k]
    q = pad_to_multiple(q, block_length)
    q = reshape_by_blocks(q, t2t_common.shape_list(q), block_length)
    total_query_blocks = t2t_common.shape_list(q)[2]

    # Set up key and value blocks.
    # [batch, heads, blocks_k, block_length, depth_k]
    blocks_per_filter_width = filter_width // block_length
    remaining_items = filter_width % block_length
    k = pad_to_multiple(k, block_length)
    v = pad_to_multiple(v, block_length)
    k = pad_l_and_r(k, filter_width + block_length - remaining_items)
    v = pad_l_and_r(v, filter_width + block_length - remaining_items)
    k = reshape_by_blocks(k, t2t_common.shape_list(k), block_length)
    v = reshape_by_blocks(v, t2t_common.shape_list(v), block_length)

    total_kv_blocks = t2t_common.shape_list(k)[2]

    slices = []
    # prepare the left-most and right-most partial blocks if needed
    if remaining_items:
        first_partial_block_k = tf.slice(
            k, [0, 0, 0, block_length - remaining_items, 0],
            [-1, -1, total_query_blocks, -1, -1])
        first_partial_block_v = tf.slice(
            v, [0, 0, 0, block_length - remaining_items, 0],
            [-1, -1, total_query_blocks, -1, -1])
        last_partial_block_k = tf.slice(
            k, [0, 0, total_kv_blocks - total_query_blocks, 0, 0],
            [-1, -1, -1, remaining_items, -1])
        last_partial_block_v = tf.slice(
            v, [0, 0, total_kv_blocks - total_query_blocks, 0, 0],
            [-1, -1, -1, remaining_items, -1])
        slices.append((first_partial_block_k, first_partial_block_v))
        slices.append((last_partial_block_k, last_partial_block_v))

    # Prepare the rest of the blocks
    first_block_index = 1 if remaining_items else 0
    attention_blocks = 2 * blocks_per_filter_width + 1
    for i in range(first_block_index, attention_blocks + first_block_index):
        block_k = tf.slice(k, [0, 0, i, 0, 0],
                           [-1, -1, total_query_blocks, -1, -1])
        block_v = tf.slice(v, [0, 0, i, 0, 0],
                           [-1, -1, total_query_blocks, -1, -1])
        slices.append((block_k, block_v))
    # [batch, heads, blocks_q, block_length + 2 * filter_width, depth_k]
    k = tf.concat([s[0] for s in slices], axis=3)
    v = tf.concat([s[1] for s in slices], axis=3)

    attention_bias = tf.expand_dims(embedding_to_padding(k) * -1e9, axis=-2)
    depth_v = t2t_common.shape_list(v)[-1]

    output, _w_dict = dot_product_attention(
        q,
        k,
        v,
        bias=attention_bias,
        dropout_rate=0.,
        name="local_1d",
        scaled=scaled)
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])

    # Remove the padding if introduced.
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape([None if isinstance(dim, tf.Tensor) else dim for dim in
                      (batch_size, num_heads, original_length, depth_v)])
    return output, _w_dict


def _make_local_block(x, depth, batch, heads, num_blocks, block_length):
    """Helper function to create a local version of the keys or values for 1d."""
    prev_block = tf.slice(x, [0, 0, 0, 0, 0],
                          [-1, -1, num_blocks - 1, -1, -1])
    cur_block = tf.slice(x, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
    local_block = tf.concat([prev_block, cur_block], 3)
    return tf.reshape(local_block,
                      [batch, heads, num_blocks - 1, block_length * 2, depth])


def masked_local_attention_1d(q,
                              k,
                              v,
                              mask=None,
                              causality_tensor=None,
                              mask_right=True,
                              block_length=128,
                              dropout_rate=0.,
                              name='masked_local_attention_1d',
                              scaled=False):
    """Attention to the source position and a neighborhood to the left of it.
    The sequence is divided into blocks of length block_length. Attention for a
    given query position can only see memory positions less than or equal to the
    query position, in the corresponding block and the previous block.
    Args:
      q: a Tensor with shape [batch, heads, length, depth_k]
      k: a Tensor with shape [batch, heads, length, depth_k]
      v: a Tensor with shape [batch, heads, length, depth_v]
      mask: optional mask of shape [batch, length] where 1 = attendable
      causality_tensor: optional tensor of shape [1/batch, length] enumerating position
      mask_right: creates right-masking if mask is not None if set to True (default: True)
      block_length: an integer
      dropout_rate: Dropout rate for attention dropout
      name: an optional string
    Returns:
      a Tensor of shape [batch, heads, length, depth_v]

    NOTE:
        USE OF MASK and CAUSALITY TENSOR IS EXPERIMENTAL!
    """

    batch, heads, length, depth_k = t2t_common.shape_list(q)
    _, heads_kv, _, _ = t2t_common.shape_list(k)
    depth_v = t2t_common.shape_list(v)[-1]
    if isinstance(block_length, tf.Tensor):
        const = tf.get_static_value(block_length)
        if const is not None:
            block_length = int(const)
    # If (length < 2 * block_length), then we use only one block.
    if isinstance(length, int) and isinstance(block_length, int):
        block_length = length if length < block_length * 2 else block_length
    else:
        block_length = tf.where(
            tf.less(length, block_length * 2), length, block_length)
    ct_batch = -1
    if causality_tensor is not None:
        ct_batch, _ = t2t_common.shape_list(causality_tensor)
    if causality_tensor is None and mask_right and mask is not None:
        causality_tensor = tf.range(0, length)
        causality_tensor = tf.reshape(causality_tensor, [1, length])
        ct_batch = 1
    # Pad query, key, value to ensure multiple of block length.
    original_length = length
    padding_size = tf.math.mod(-length, block_length)
    length += padding_size
    padding = [[0, 0], [0, 0], [0, padding_size], [0, 0]]
    q = tf.pad(q, padding)
    k = tf.pad(k, padding)
    v = tf.pad(v, padding)
    if mask is not None:
        # tf.get_logger().warning('Use of mask and/or causality tensor in masked_local_attention_1d is experimental')
        mask = tf.reshape(mask, [batch, 1, original_length, 1])
        mask = tf.pad(mask, padding)
        if mask_right or causality_tensor is not None:
            causality_tensor = tf.reshape(causality_tensor, [ct_batch, 1, original_length, 1])
            causality_tensor = tf.pad(causality_tensor, padding, constant_values=original_length + 1)
    if isinstance(length, int) and isinstance(block_length, int):
        num_blocks = length // block_length
    else:
        num_blocks = tf.cast(tf.math.divide(length, block_length), tf.int32)

    # Compute attention for the first query block.
    first_q = tf.slice(q, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_k = tf.slice(k, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_v = tf.slice(v, [0, 0, 0, 0], [-1, -1, block_length, -1])
    if mask is not None:
        first_mask = tf.slice(mask, [0, 0, 0, 0], [-1, -1, block_length, -1])
        if mask_right or causality_tensor is not None:
            first_ct = tf.slice(causality_tensor, [0, 0, 0, 0], [-1, -1, block_length, -1])
            first_ct = tf.cast(tf.linalg.matrix_transpose(first_ct) <= first_ct, mask.dtype)
            first_mask = tf.minimum(first_mask, first_ct)
        first_bias = -1e9 * (1.0 - first_mask)
    else:
        first_bias = attention_bias_lower_triangle(block_length)

    first_output, first_dict = dot_product_attention(
        first_q,
        first_k,
        first_v,
        bias=first_bias,
        dropout_rate=dropout_rate,
        name="first_block",
        scaled=scaled)

    # Compute attention for all subsequent query blocks.
    q = tf.reshape(q, [batch, heads, num_blocks, block_length, depth_k])
    k = tf.reshape(k, [batch, heads_kv, num_blocks, block_length, depth_k])
    v = tf.reshape(v, [batch, heads_kv, num_blocks, block_length, depth_v])
    if mask is not None:
        mask = tf.reshape(mask, [batch, 1, num_blocks, block_length, 1])
        local_mask = _make_local_block(mask, 1, batch, 1, num_blocks,
                                       block_length)
        tail_mask = tf.slice(mask, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
        tail_mask = tf.reshape(tail_mask,
                               [batch, 1, num_blocks - 1, block_length, 1])
        local_mask = tf.minimum(tf.linalg.matrix_transpose(local_mask), tail_mask)
        if causality_tensor is not None:
            ct = tf.reshape(causality_tensor, [ct_batch, 1, num_blocks, block_length, 1])
            local_ct = _make_local_block(ct, 1, ct_batch, 1, num_blocks,
                                         block_length)
            tail_ct = tf.slice(ct, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
            tail_ct = tf.reshape(tail_ct, [ct_batch, 1, num_blocks - 1, block_length, 1])
            ct_mask = tf.cast(tf.linalg.matrix_transpose(local_ct) <= tail_ct, local_mask.dtype)
            local_mask = tf.minimum(local_mask, ct_mask)

    local_k = _make_local_block(k, depth_k, batch, heads_kv, num_blocks,
                                block_length)
    local_v = _make_local_block(v, depth_v, batch, heads_kv, num_blocks,
                                block_length)
    tail_q = tf.slice(q, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
    tail_q = tf.reshape(tail_q,
                        [batch, heads, num_blocks - 1, block_length, depth_k])
    local_length = t2t_common.shape_list(local_k)[3]

    # make sure source_pos <= target_pos
    good_part = t2t_common.ones_matrix_band_part(
        block_length,
        local_length,
        -1,
        block_length,
        out_shape=[1, 1, 1, block_length, local_length])
    if mask is not None:
        bias = (1.0 - local_mask) * -1e9
    else:
        bias = (1.0 - good_part) * -1e9

    # TODO(noam): figure out how to show a summary for the remaining blocks.
    # The naive way currently causes errors due to empty tensors.
    # output: [batch, heads, num_blocks-1, block_length, depth_v]
    tail_output, tail_dict = dot_product_attention(
        tail_q,
        local_k,
        local_v,
        bias=bias,
        dropout_rate=dropout_rate,
        name="tail_block",
        scaled=scaled)
    tail_output = tf.reshape(
        tail_output, [batch, heads, (num_blocks - 1) * block_length, depth_v])
    output = tf.concat([first_output, tail_output], axis=2)

    # Remove the padding if introduced.
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output = tf.reshape(output, [batch, heads, original_length, depth_v])
    return output, {**first_dict, **tail_dict}


def combine_first_two_dimensions(x):
    """Reshape x so that the first two dimension become one.
    Args:
      x: a Tensor with shape [a, b, ...]
    Returns:
      a Tensor with shape [ab, ...]
    """
    ret = tf.reshape(x, tf.concat([[-1], t2t_common.shape_list(x)[2:]], 0))
    old_shape = x.get_shape().dims
    a, b = old_shape[:2]
    new_shape = [a * b if a and b else None] + old_shape[2:]
    ret.set_shape(new_shape)
    return ret


def dot_product_batched_head(q, k, v, gates_q, gates_k, mask_right=False, scaled=False):
    """Perform a dot product attention on a single sequence on a single head.
    This function dispatch the q, k, v and loop over the buckets to compute the
    attention dot product on each subsequences.
    Args:
      q (tf.Tensor): [batch*heads, length_q, depth_q]
      k (tf.Tensor): [batch*heads, length_kv, depth_q]
      v (tf.Tensor): [batch*heads, length_kv, depth_v]
      gates_q (tf.Tensor): One-hot of shape [batch*heads, length_q, nb_buckets]
      gates_k (tf.Tensor): One-hot of shape [batch*heads, length_k, nb_buckets]
      mask_right (bool): Add a bias to prevent attention to the future
    Returns:
      tf.Tensor: [length_q, depth_v]
    """
    nb_buckets = t2t_common.shape_list(gates_q)[-1]

    def get_dispatcher(gates):
        """Construct dispatcher for gates."""
        length = t2t_common.shape_list(gates)[1]
        # Count the number of ones per batch (and keep the max value)
        nb_elems_to_dispatch = tf.reduce_sum(gates, axis=[1, 2])
        nb_elems_to_dispatch = tf.reduce_max(nb_elems_to_dispatch)
        nb_elems_to_dispatch = to_int32(nb_elems_to_dispatch)
        capacity = nb_elems_to_dispatch // nb_buckets * 2  # Capacity is hardcoded
        capacity = tf.minimum(length, capacity)
        capacity = tf.maximum(capacity, 1)
        return t2t_expert_util.TruncatingDispatcher(gates, capacity)

    q_dispatcher = get_dispatcher(gates_q)
    k_dispatcher = get_dispatcher(gates_k)

    q = q_dispatcher.dispatch(q)
    k = k_dispatcher.dispatch(k)
    v = k_dispatcher.dispatch(v)

    # Bias of shape [batch*heads, nb_buckets, 1, capacity] broadcasted to every
    # queries
    bias = tf.expand_dims((k_dispatcher.nonpadding() - 1.0) * 1e9, 2)
    if mask_right:
        q_coordinate = t2t_common.to_float(
            tf.expand_dims(q_dispatcher.length_coordinate(), 3))
        k_coordinate = t2t_common.to_float(
            tf.expand_dims(k_dispatcher.length_coordinate(), 2))
        bias += t2t_common.to_float(tf.greater(k_coordinate, q_coordinate)) * -1e9
    # The sequence padding is not masked but is ignored on the next layers

    # q, k, v now have shape [batch*heads, nb_bucket, capacity, depth]
    # The buckets can be seen as different heads
    v_out, w_dict = dot_product_attention(q, k, v, bias=bias, scaled=scaled)

    # Combine all buckets together to restore the original length
    return q_dispatcher.combine(v_out), w_dict


def sparse_dot_product_attention_truncated(
        q,
        k,
        v,
        list_lsh,
        mask_right=False,
        scaled=False):  # pylint: disable=unused-argument
    """Sparse multihead self attention.
    Perform an approximation of the full multihead attention by dispatching
    the tokens using their keys/values. Thus the attention matrix are only
    computed each times on a subset of the tokens.
    Notes:
     * The function don't perform scaling here (multihead_attention does
    the /sqrt(depth)).
     * The padding should have been removed (so batch size should be 1 but length
     contains the elements from all different batches)
     * Right now, only self attention is supported so length_q and length_kv
     should be identical and the function will add triangular bias.
     * If bi.order is not None, The bias is added inside this function to
     prevent attention to the future.
    Args:
      q (tf.Tensor): Queries of shape [batch, heads, length_q, depth_k]
      k (tf.Tensor): Keys of shape [batch, heads, length_kv, depth_k]
      v (tf.Tensor): Values of shape [batch, heads, length_kv, depth_v]
      list_lsh: List of layers that can perform lsh-gating
      mask_right (bool):
    Returns:
      tf.Tensor: Approximation of Softmax(Q.K) * V, of shape
        [batch, heads, length_q, depth_v]
    """
    # Currently depth is the same for for q and v
    batch_size, nb_heads, _, depth = t2t_common.shape_list(q)
    _, nb_heads_kv, _, _ = t2t_common.shape_list(k)

    total_loss = 0.0

    # Each head get its own dispatcher
    # list_lsh = [LshGating(depth=depth, **experts_params) for _ in range(nb_heads)]

    def get_gates_head(x, add_first=False):
        """Return the gates for each heads of the current x.
        Args:
          x (tf.Tensor): of shape [batch, heads, length, depth]
          add_first (bool): if True, add the first element on each bucket
        Returns:
          tf.Tensor: gates of shape [batch, heads, length, num_buckets]
        """
        length = t2t_common.shape_list(x)[2]

        # Invert heads/batch
        x = tf.transpose(x, perm=[1, 0, 2, 3])
        x = tf.reshape(x, [nb_heads, batch_size * length, depth])

        list_x = tf.unstack(x)  # list[tf.Tensor(shape=[batch * length, depth])]

        # Unstack heads
        list_gates = []
        # There might be a more optimized way to compute all heads at once
        for lsh, single_x in zip(list_lsh, list_x):
            # Each head get its own dispatcher
            gates = lsh(single_x)
            nb_buckets = gates.get_shape().as_list()[-1]
            # Reshape to [batch, length, depth] but should consider sequence
            # padding in that case (also dispatch the padding)
            gates = tf.reshape(gates, [batch_size, length, nb_buckets])
            list_gates.append(gates)

        gates = tf.stack(list_gates)

        # Restore original shape
        gates = tf.reshape(gates, [nb_heads, batch_size, length, nb_buckets])
        gates = tf.transpose(gates, [1, 0, 2, 3])

        # Dispatch the first element to every gates to avoid empty buckets
        if add_first:
            gates = tf.maximum(gates,
                               tf.reshape(tf.one_hot([0], length), [1, 1, length, 1]))

        return gates

    gates_q = get_gates_head(q)
    gates_k = get_gates_head(k, add_first=True)

    # [batch, heads, length, depth] => [batch*heads, length, depth]
    q, k, v, gates_q, gates_k = [
        combine_first_two_dimensions(t) for t in (q, k, v, gates_q, gates_k)
    ]

    v_out, w_dict, = dot_product_batched_head(q, k, v, gates_q, gates_k, mask_right, scaled=scaled)

    # Restore original dimension
    v_out = tf.reshape(v_out, [batch_size, nb_heads, -1, depth])

    return v_out, total_loss / nb_heads, w_dict


def _matmul_with_relative_keys_2d(x, y, heads_share_relative_embedding):
    """Helper function for dot_product_unmasked_self_attention_relative_2d."""
    if heads_share_relative_embedding:
        ret = tf.einsum("bhxyd,md->bhxym", x, y)
    else:
        ret = tf.einsum("bhxyd,hmd->bhxym", x, y)
    return ret


def dot_product_unmasked_self_attention_relative_2d(
        q, k, v, bias,
        height_key_relative_embeddings,
        width_key_relative_embeddings,
        max_relative_position=None, dropout_rate=0.0, name=None,
        dropout_broadcast_dims=[0, 1], heads_share_relative_embedding=False,
        add_relative_to_values=False):
    """Calculate relative position unmasked dot-product self-attention 2d.
    The attention calculation is augmented with learned representations for the
    relative position between each element in q and each element in k and v in
    height and width dimensions. for query index (i,j) and key index (l, m),
    the logit is q_i k_j^T + q_i rh_{l-i}^T + q_i rw_{m-j}^T, where rh and ry are
    the set of relative embeddings in height and width spatial dimensions,
    respectively.
    Args:
      q: a Tensor with shape [batch, heads, height, width, depth].
      k: a Tensor with shape [batch, heads, height, width, depth].
      v: a Tensor with shape [batch, heads, height, width, depth].
      bias: bias Tensor.
      max_relative_position: an integer the max relative embedding considered.
        Changing this invalidates checkpoints.
      dropout_rate: a floating point number.
      image_shapes: optional tuple of integer scalars.
      name: an optional string.
      make_image_summary: Whether to make an attention image summary.
      dropout_broadcast_dims:  an optional list of integers less than 4
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.
      heads_share_relative_embedding: a boolean indicating wheather to share
        relative embeddings between attention heads.
      add_relative_to_values: a boolean for adding relative embeddings to values.
    Returns:
      [batch, heads, height, width, depth] tensor, the output of attention.
      height_key_relative_embeddings: a 3d or 2d tensor, depending on head sharing
        settings, which are the relative embeddings for height.
      width_key_relative_embeddings: a 3d or 2d tensor, depending on head sharing
        settings, which are the relative embeddings for width.
    Raises:
      ValueError: if max_relative_position is not > 0.
    """
    if not max_relative_position:
        raise ValueError("Max relative position (%s) should be > 0 when using "
                         "relative self attention." % (max_relative_position))

    if add_relative_to_values:
        raise ValueError("Adding relative embeddings to values is not implemented")

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    q.get_shape()[2:-1].assert_is_compatible_with(k.get_shape()[2:-1])
    q.get_shape()[2:-1].assert_is_compatible_with(v.get_shape()[2:-1])

    (height, width) = (t2t_common.shape_list(q)[2],
                       t2t_common.shape_list(q)[3])
    q_shape = t2t_common.shape_list(q)
    k_shape = t2t_common.shape_list(k)
    num_heads_q = q_shape[1]
    num_heads_k = k_shape[1]
    mqa = num_heads_q > num_heads_k
    depth_k = k_shape[-1]
    depth_v = t2t_common.shape_list(v)[-1]
    # flatten height width
    flatten_hw = lambda x, d, num_heads: tf.reshape(x, [-1, num_heads, height * width, d])
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(flatten_hw(q, depth_k, num_heads_q), flatten_hw(k, depth_k, num_heads_k),
                       transpose_b=True)

    def _compute_2d_relative_logits(
            query, key_relative_embeddings, height, width,
            heads_share_relative_embedding, transpose_mask, num_heads):
        """compute relative logits."""
        unmasked_rel_logits = _matmul_with_relative_keys_2d(
            query, key_relative_embeddings, heads_share_relative_embedding)
        # collapse height and heads
        unmasked_rel_logits = tf.reshape(unmasked_rel_logits,
                                         [-1, num_heads * height, width,
                                          2 * width - 1])
        unmasked_rel_logits = (
            _relative_position_to_absolute_position_unmasked(
                unmasked_rel_logits))
        # shape it back for tiling
        unmasked_rel_logits = tf.reshape(
            unmasked_rel_logits, [-1, num_heads, height, width, width])
        # tiling it height times
        unmasked_rel_logits = tf.expand_dims(
            unmasked_rel_logits, axis=3)
        unmasked_rel_logits = tf.tile(unmasked_rel_logits,
                                      [1, 1, 1, height, 1, 1])
        # bringing it to the right shape for adding to the logits.
        unmasked_rel_logits = tf.transpose(unmasked_rel_logits, transpose_mask)
        unmasked_rel_logits = tf.reshape(unmasked_rel_logits,
                                         [-1, num_heads, height * width,
                                          height * width])
        return unmasked_rel_logits

    # Relative logits in width dimension first.
    width_key_relative_embeddings = get_relative_embeddings_left_right(
        width_key_relative_embeddings,
        max_relative_position, width, depth_k,
        heads_share_relative_embedding)
    # [batch, heads, height, 2*width-1, 2*width-1]
    width_unmasked_rel_logits = _compute_2d_relative_logits(
        q, width_key_relative_embeddings, height, width,
        heads_share_relative_embedding, [0, 1, 2, 4, 3, 5], num_heads=num_heads_q)
    logits += width_unmasked_rel_logits
    # Relative logits in height dimension next. For ease, we transpose
    # height and width and repeat the above steps, and transpose to eventually
    # put the logits in their right positions.
    # [batch, heads, height, 2*height-1, 2*width-1]
    height_key_relative_embeddings = get_relative_embeddings_left_right(
        height_key_relative_embeddings,
        max_relative_position, height, depth_k,
        heads_share_relative_embedding)

    height_unmasked_rel_logits = _compute_2d_relative_logits(
        tf.transpose(q, [0, 1, 3, 2, 4]),
        height_key_relative_embeddings,
        width,
        height,
        heads_share_relative_embedding, [0, 1, 4, 2, 5, 3], num_heads=num_heads_q)
    logits += height_unmasked_rel_logits
    if bias is not None:
        logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = t2t_common.dropout_with_broadcast_dims(
        weights, dropout_rate, broadcast_dims=dropout_broadcast_dims)

    ret = tf.matmul(weights, flatten_hw(v, depth_v, num_heads_k))
    # reshape back the same spatial dimensions as q
    return tf.reshape(ret, [-1, num_heads_q, height, width, depth_v]), weights


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
      length: scalar, length of timing signal sequence.
      channels: scalar, size of timing embeddings to create. The number of
          different timescales is equal to channels / 2.
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position
    Returns:
      a Tensor of timing signals [1, length, channels]
    """
    position = t2t_common.to_float(tf.range(1, 1 + length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            tf.maximum(t2t_common.to_float(num_timescales) - 1, 1))
    inv_timescales = min_timescale * tf.exp(
        t2t_common.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    # Please note that this slightly differs from the published paper.
    # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.math.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def add_timing_signal_1d(x,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
      x: a Tensor with shape [batch, length, channels]
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position
    Returns:
      a Tensor the same shape as x.
    """
    length = t2t_common.shape_list(x)[1]
    channels = t2t_common.shape_list(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale,
                                  start_index)
    return x + t2t_common.cast_like(signal, x)

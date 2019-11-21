import tensorflow as tf

from dynastes.ops.attention_nd import split_heads, scaled_dot_product_attention, merge_heads
from dynastes.ops.pad_ops import pad_input_2d


def extract_and_split_2d(x, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same'):
    shape = x.shape.as_list()
    x, padding = pad_input_2d(x, padding, kernel_size=kernel_size, dilation_rate=dilation_rate)
    x = tf.image.extract_patches(
        x,
        [1, kernel_size[0], kernel_size[1], 1],
        [1, strides[0], strides[1], 1],
        [1, dilation_rate[0], dilation_rate[1], 1],
        padding=padding)
    shape_p = x.shape.as_list()
    x = tf.reshape(x, shape=shape_p[:-1] + [kernel_size[0] * kernel_size[1], shape[-1]])
    return x


def localized_attention_2d(q, k, v, num_heads=1,
                           kernel_size=(2, 2),
                           strides=(1, 1),
                           dilation_rate=(1, 1),
                           padding='same',
                           preshaped_q=True,
                           multiquery_attention=False):
    xk = extract_and_split_2d(k,
                              kernel_size=kernel_size,
                              strides=strides,
                              dilation_rate=dilation_rate,
                              padding=padding)
    if not multiquery_attention:
        xk = split_heads(xk, num_heads, ignore_dims=2)
    xv = extract_and_split_2d(v,
                              kernel_size=kernel_size,
                              strides=strides,
                              dilation_rate=dilation_rate,
                              padding=padding)
    if not multiquery_attention:
        xv = split_heads(xv, num_heads, ignore_dims=2)

    if preshaped_q:
        xq = split_heads(q, num_heads=num_heads, ignore_dims=2)
        xq = tf.expand_dims(xq, -2)
    else:
        xq = extract_and_split_2d(q,
                                  kernel_size=(1, 1),
                                  strides=strides,
                                  dilation_rate=(1, 1),
                                  padding=padding)
        xq = split_heads(xq, num_heads=num_heads, ignore_dims=2)

    r, _ = scaled_dot_product_attention(q=xq, k=xk, v=xv, bias=None, multiquery_attention=multiquery_attention)

    x = merge_heads(r, 2)
    x = tf.squeeze(x, -2)

    return x


def localized_attention_1d(q, k, v, num_heads=1,
                           kernel_size=2,
                           strides=1,
                           dilation_rate=1,
                           padding='same',
                           preshaped_q=True,
                           multiquery_attention=False):
    q = tf.expand_dims(q, 1)
    k = tf.expand_dims(k, 1)
    v = tf.expand_dims(v, 1)

    x = localized_attention_2d(q, k, v, num_heads=num_heads,
                               kernel_size=(1, kernel_size),
                               strides=(1, strides),
                               dilation_rate=(1, dilation_rate),
                               padding=padding,
                               preshaped_q=preshaped_q,
                               multiquery_attention=multiquery_attention)

    x = tf.squeeze(x, 1)
    return x

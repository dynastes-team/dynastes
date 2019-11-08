import tensorflow as tf

from dynastes.ops.attention import split_heads, scaled_dot_product_attention, merge_heads


def extract_and_split_2d(x, num_heads, patch_size=(3, 3), strides=(1, 1), dilations=(1, 1), padding='SAME'):
    shape = x.shape.as_list()
    x = tf.image.extract_patches(
        x,
        [1, patch_size[0], patch_size[1], 1],
        [1, strides[0], strides[1], 1],
        [1, dilations[0], dilations[1], 1],
        padding=padding)
    shape_p = x.shape.as_list()
    x = tf.reshape(x, shape=shape_p[:-1] + [patch_size[0] * patch_size[1], shape[-1]])

    return split_heads(x, num_heads, ignore_dims=2)


def localized_attention_2d(q, k, v, num_heads=1,
                           patch_size=(2, 2),
                           strides=(1, 1),
                           dilations=(1, 1),
                           padding='same',
                           preshaped_q=True):

    xk = extract_and_split_2d(k,
                              num_heads=num_heads,
                              patch_size=patch_size,
                              strides=strides,
                              dilations=dilations,
                              padding=padding.upper())

    xv = extract_and_split_2d(v, num_heads=num_heads,
                              patch_size=patch_size,
                              strides=strides,
                              dilations=dilations,
                              padding=padding.upper())

    if preshaped_q:
        xq = split_heads(q, num_heads=num_heads, ignore_dims=2)
        xq = tf.expand_dims(xq, -2)
    else:
        xq = extract_and_split_2d(q,
                                  num_heads=num_heads,
                                  patch_size=(1, 1),
                                  strides=strides,
                                  dilations=(1, 1),
                                  padding=padding.upper())

    r, _ = scaled_dot_product_attention(q=xq, k=xk, v=xv, mask=None)

    x = merge_heads(r, 2)
    x = tf.squeeze(x, -2)

    return x


def localized_attention_1d(q, k, v, num_heads=1, patch_size=2, stride=1, dilation=1, padding='same', preshaped_q=True):

    q = tf.expand_dims(q, 1)
    k = tf.expand_dims(k, 1)
    v = tf.expand_dims(v, 1)

    xk = extract_and_split_2d(k,
                              num_heads=num_heads,
                              patch_size=(1, patch_size),
                              strides=(1, stride),
                              dilations=(1, dilation),
                              padding=padding.upper())

    xv = extract_and_split_2d(v,
                              num_heads=num_heads,
                              patch_size=(1, patch_size),
                              strides=(1, stride),
                              dilations=(1, dilation),
                              padding=padding.upper())

    if preshaped_q:
        xq = split_heads(q, num_heads=num_heads, ignore_dims=2)
        xq = tf.expand_dims(xq, -2)
    else:
        xq = extract_and_split_2d(q,
                                  num_heads=num_heads,
                                  patch_size=(1, 1),
                                  strides=(1, stride),
                                  dilations=(1, dilation),
                                  padding=padding.upper())

    r, _ = scaled_dot_product_attention(q=xq, k=xk, v=xv, mask=None)

    x = merge_heads(r, 2)
    x = tf.squeeze(x, -2)

    x = tf.squeeze(x, 1)
    return x
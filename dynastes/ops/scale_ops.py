import tensorflow as tf

from dynastes.ops.t2t_common import shape_list


def _upscale2d(x, strides, method, antialias=True, gain=1):
    x_shape = shape_list(x)
    ret_h = x_shape[1] * strides[0]
    ret_w = x_shape[2] * strides[1]

    # Apply gain.
    if gain != 1:
        x *= gain

    return tf.image.resize(x, size=[ret_h, ret_w], method=method, antialias=antialias)


def _downscale2d(x, strides, method, antialias=True, gain=1):
    x_shape = shape_list(x)
    ret_h = x_shape[1] // strides[0]
    ret_w = x_shape[2] // strides[1]

    # Apply gain.
    if gain != 1:
        x *= gain

    return tf.image.resize(x, size=[ret_h, ret_w], method=method, antialias=antialias)


def upscale2d(x, strides=(2, 2), method='bilinear', antialias=True):
    with tf.name_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, strides=strides, method=method, antialias=antialias)

            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, strides=strides, method=method, antialias=antialias,
                                  gain=((strides[0] + strides[1]) / 2) ** 2)
                return dx, lambda ddx: _upscale2d(ddx, strides=strides, method=method, antialias=antialias)

            return y, grad

        return func(x)


def downscale2d(x, strides=(2, 2), method='bilinear', antialias=True):
    with tf.name_scope('Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, strides=strides, method=method, antialias=antialias)

            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, strides=strides, method=method, antialias=antialias,
                                gain=1 / ((strides[0] + strides[1]) / 2) ** 2)
                return dx, lambda ddx: _downscale2d(ddx, strides=strides, method=method, antialias=antialias)

            return y, grad

        return func(x)

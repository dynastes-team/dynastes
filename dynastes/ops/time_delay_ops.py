import tensorflow as tf

from dynastes.ops.pad_ops import pad_input_2d


def time_delay_nn_1d(x, kernel, kernel_size, strides, dilation_rate, padding='valid'):
    shape = x.shape.as_list()
    x = tf.expand_dims(x, -1)
    x, padding = pad_input_2d(x, padding, kernel_size=(kernel_size, shape[-1]), dilation_rate=(dilation_rate, 1))
    x = tf.image.extract_patches(x,
                                 sizes=[1, kernel_size, shape[-1], 1],
                                 strides=[1, strides, shape[-1], 1],
                                 rates=[1, dilation_rate, 1, 1],
                                 padding=padding)
    x = tf.squeeze(x, -2)
    x = tf.matmul(x, kernel)
    return x

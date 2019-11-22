import tensorflow as tf


def crop(x, crops):
    crops = tf.convert_to_tensor(crops)
    begins, ends = tf.unstack(crops, axis=-1)
    shape = tf.convert_to_tensor(x.shape.as_list())
    return tf.slice(x, begins, shape - (ends+begins))

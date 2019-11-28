import tensorflow as tf

from dynastes.ops.t2t_common import shape_list


def crop(x, crops):
    crops = tf.convert_to_tensor(crops)
    begins, ends = tf.unstack(crops, axis=-1)
    shape = tf.convert_to_tensor(shape_list(x))
    return tf.slice(x, begins, shape - (ends + begins))

import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer


class Orthogonal(Regularizer):
    """Regularizer base class.
    """

    def __init__(self, scale=0.0001):
        self.scale = scale

    def __call__(self, w):
        shape = w.get_shape().as_list()
        c = shape[-1]
        if len(shape) > 2:
            w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return self.scale * ortho_loss

    def get_config(self):
        return {'scale': float(self.scale)}

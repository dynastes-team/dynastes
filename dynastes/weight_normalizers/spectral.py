import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class SpectralNormalization(tfkl.Layer):
    def __init__(self,
                 power_iteration_rounds=1,
                 equality_constrained=True,
                 transposed=False,
                 trainable=None,
                 **kwargs):
        super(SpectralNormalization, self).__init__(**kwargs, trainable=False)
        self.power_iteration_rounds = power_iteration_rounds
        self.equality_constrained = equality_constrained
        self.transposed = transposed

    def build(self, input_shape):
        if self.transposed:
            input_shape = input_shape[:-2] + [input_shape[-1], input_shape[-2]]
        u_shape = (np.cumprod(np.array(input_shape[:-1]))[-1], 1)

        replica_context = tf.distribute.get_replica_context()
        if replica_context is None:  # cross repica strategy.
            # TODO(joelshor): Determine appropriate aggregation method.
            # https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/features/spectral_normalization.py
            raise ValueError("spectral norm isn't supported in cross-replica "
                             "distribution strategy.")
        elif not tf.distribute.has_strategy():  # default strategy.
            aggregation = None
        else:
            aggregation = tf.VariableAggregation.ONLY_FIRST_REPLICA

        self.u = self.add_weight(name='u',
                                 shape=u_shape,
                                 dtype=self.dtype,
                                 initializer=tfk.initializers.RandomNormal(),
                                 trainable=False,
                                 aggregation=aggregation)
        if self.transposed:
            input_shape = input_shape[:-2] + [input_shape[-1], input_shape[-2]]
        super(SpectralNormalization, self).build(input_shape)

    def _compute_spectral_norm(self, w, training=None):
        w = tf.reshape(w, (-1, w.get_shape()[-1]))
        u = self.u
        # Use power iteration method to approximate spectral norm.
        for i in range(self.power_iteration_rounds):
            # `v` approximates the first right singular vector of matrix `w`.
            v = tf.nn.l2_normalize(tf.matmul(a=w, b=u, transpose_a=True))
            u = tf.nn.l2_normalize(tf.matmul(w, v))

        # Update persisted approximation.
        if training:
            with tf.control_dependencies([self.u.assign(u, name='update_u')]):
                u = tf.identity(u)

        u = tf.stop_gradient(u)
        v = tf.stop_gradient(v)

        # Largest singular value of `w`.
        spectral_norm = tf.matmul(tf.matmul(a=u, b=w, transpose_a=True), v)
        spectral_norm.shape.assert_is_fully_defined()
        spectral_norm.shape.assert_is_compatible_with([1, 1])

        return spectral_norm[0][0]

    def call(self, w, training=None):
        if self.transposed:
            w = tf.linalg.matrix_transpose(w)
        normalization_factor = self._compute_spectral_norm(w, training=training)
        if not self.equality_constrained:
            normalization_factor = tf.maximum(1., normalization_factor)
        w_normalized = w / normalization_factor
        w_normalized = tf.reshape(w_normalized, w.get_shape())
        if self.transposed:
            w_normalized = tf.linalg.matrix_transpose(w_normalized)
        return w_normalized

    def get_config(self):
        config = {
            'power_iteration_rounds': self.power_iteration_rounds,
            'equality_constrained': self.equality_constrained,
            'transposed': self.transposed
        }
        base_config = super(SpectralNormalization, self).get_config()
        return {**base_config, **config}

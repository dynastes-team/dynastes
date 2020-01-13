import tensorflow as tf
import tensorflow.keras.layers as tfkl

from dynastes.ops.t2t_common import shape_list


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class StatelessRandomNormalLike(tfkl.Layer):
    """
    A layer normal producing layer that outputs shape like input
    with optional seed of lower frequency, ie:
    inputs[0] = tf.Tensor of shape [b, x, y, ch]
    inputs[1] = tf.Tensor (int32/int64) of shape [b, x//4]

    Useful for cases where you need deterministic random
    such as inverse-autoregressive flows or StyleGAN noise
    """

    def __init__(self,
                 mean=0.0,
                 stddev=1.0,
                 channels=-1,
                 **kwargs):
        super(StatelessRandomNormalLike, self).__init__(**kwargs)
        self.mean = mean
        self.stddev = stddev
        self.channels = channels

    def build(self, input_shape):
        super(StatelessRandomNormalLike, self).build(input_shape)

    def _compute_shape(self, inputs, seed):
        inputs_shape = shape_list(inputs)
        if self.channels > 0:
            inputs_shape[-1] = self.channels
        seed_shape = shape_list(seed)
        seed_shape = seed_shape + ([1] * (len(inputs_shape) - len(seed_shape)))
        random_shape = [a // b for a, b in zip(inputs_shape, seed_shape)]
        while random_shape[0] == 1:
            random_shape = random_shape[1:]
        return tf.stack(random_shape)

    def _recurse_generate(self, seed, shape):
        seed_shape = shape_list(seed)
        if len(seed_shape) == 0:
            if seed.dtype == tf.int64:
                seed = tf.bitcast(seed, tf.int32)
            elif seed.dtype == tf.int32:
                seed = tf.bitcast(seed, tf.int16)
                seed = tf.cast(seed, tf.int32)
            return tf.random.stateless_normal(mean=self.mean, stddev=self.stddev, seed=seed, shape=shape)
        else:
            return tf.stack([self._recurse_generate(s, shape) for s in tf.unstack(seed)])

    def _stateless_random_normal(self, inputs, seed=None):
        inputs_shape = shape_list(inputs)
        if self.channels > 0:
            inputs_shape[-1] = self.channels
        if seed is None:
            return tf.random.normal(mean=self.mean, stddev=self.stddev, shape=inputs_shape)
        else:
            random_shape = self._compute_shape(inputs, seed)
            out = self._recurse_generate(seed, random_shape)
            return tf.reshape(out, inputs_shape)

    def call(self, inputs, training=None, mask=None):
        if type(inputs) == list:
            seed = inputs[1]
            inputs = inputs[0]
        else:
            seed = None
        return self._stateless_random_normal(inputs, seed)

    def compute_output_shape(self, input_shape):
        if type(input_shape) == list:
            inputs_shape = input_shape[0]
        else:
            inputs_shape = input_shape
        if self.channels > 0:
            inputs_shape[-1] = self.channels
        return inputs_shape

    def get_config(self):
        config = {
            'mean': self.mean,
            'stddev': self.stddev,
            'channels': self.channels,
        }
        base_config = super(StatelessRandomNormalLike, self).get_config()
        return {**base_config, **config}

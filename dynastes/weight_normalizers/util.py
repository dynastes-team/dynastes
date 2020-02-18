import numpy as np
import six
import tensorflow as tf
import tensorflow.keras.initializers as tfki
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object, deserialize_keras_object
from tensorflow.python.ops import variables as tf_variables

from dynastes.ops.t2t_common import shape_list
from dynastes.weight_normalizers.spectral import SpectralNormalization


def serialize(normalizer):
    return serialize_keras_object(normalizer)


def deserialize(config, custom_objects={}):
    custom_objects = {**custom_objects, **object_scope}
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='normalizer')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        identifier = str(identifier)
        if identifier == 'spectral':
            return SpectralNormalization()
        elif identifier == 'spectral_t':
            return SpectralNormalization(transposed=True)
        elif identifier == 'wscale':
            return WscaleNormalizer()
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret normalizer identifier:', identifier)


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class WscaleNormalizer(tfkl.Layer):

    def __init__(self,
                 lrmul=1.,
                 gain=np.sqrt(2),
                 next_layer=tfkl.Activation('linear'),
                 trainable=False,
                 **kwargs):
        if 'trainable' in kwargs:
            kwargs.pop('trainable')
        super(WscaleNormalizer, self).__init__(trainable=False, **kwargs)
        self.next_layer = get(next_layer)
        self.gain = gain
        self.lrmul = lrmul

    def call(self, inputs, training=None):
        input_shape = shape_list(inputs)
        if len(input_shape) > 1:
            fan_in = np.prod(input_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
            he_std = self.gain / np.sqrt(fan_in)  # He init
            runtime_coef = he_std * self.lrmul
        else:
            runtime_coef = self.lrmul
        return self.next_layer(inputs * runtime_coef, training=training)

    def get_config(self):
        config = {
            'lrmul': self.lrmul,
            'gain': self.gain,
            'next_layer': serialize(self.next_layer)
        }
        base_config = super(WscaleNormalizer, self).get_config()
        return {**base_config, **config}


cs = tf.CriticalSection(name='init_mutex')


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class WeightNormalizer(tfkl.Layer):

    def __init__(self,
                 weight_initializer,
                 next_layer=tfkl.Activation('linear'),
                 **kwargs):
        super(WeightNormalizer, self).__init__(**kwargs)
        self.orig_weight_initializer = tfki.get(weight_initializer)
        self.next_layer = get(next_layer)
        self._init_critical_section = cs
        self.g = None

    def build(self, input_shape):
        self.layer_depth = int(input_shape[-1])
        self.kernel_norm_axes = list(range(len(input_shape) - 1))
        """
        self._initialized_g = self.add_weight(
            name='initialized_g',
            shape=None,
            initializer="zeros",
            dtype=tf.dtypes.bool,
            synchronization=tf_variables.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA,
        )"""

        # For now it's a workaround, maybe works, who knows ¯\_(ツ)_/¯

        def get_weight_init(input_shape, original_init):
            def weight_init(target_shape, dtype, input_shape, original_init):
                var_init = original_init(input_shape, dtype)
                var_norm = tf.norm(tf.reshape(var_init, [-1, target_shape[-1]]), axis=0)

                return tf.random.normal(shape=target_shape, mean=tf.reduce_mean(var_norm),
                                        stddev=tf.math.reduce_std(var_norm))

            return lambda x, dtype: weight_init(x, dtype, input_shape, original_init)

        self.g = self.add_weight(
            name="g",
            shape=(self.layer_depth,),
            synchronization=tf_variables.VariableSynchronization.AUTO,
            initializer=get_weight_init(input_shape, self.orig_weight_initializer),
            trainable=True,
        )

        self.built = True

    def call(self, inputs, training=None, **kwargs):
        # TODO: Fix this when TensorFlow developers learn how to code if I haven't switched to PyTorch by that time

        # For now it's a workaround, maybe works, who knows ¯\_(ツ)_/¯

        # def _update_or_return_vars():
        #    return tf.identity(self.g)

        # def _init_g():
        #    V_norm = tf.norm(tf.reshape(inputs, [-1, self.layer_depth]), axis=0)
        #    with tf.control_dependencies([self.g.assign(V_norm), self._initialized_g.assign(True)]):
        #        return tf.identity(self.g)

        # g = self._init_critical_section.execute(lambda: tf.cond(self._initialized_g, _update_or_return_vars, _init_g))
        g = self.g

        V_norm = tf.norm(tf.reshape(inputs, [-1, self.layer_depth]), axis=0)
        scaler = tf.reshape(tf.math.divide_no_nan(g, V_norm),
                            list([1] * len(self.kernel_norm_axes)) + [self.layer_depth])
        return inputs * scaler

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'weight_initializer': tfki.serialize(self.orig_weight_initializer),
            'next_layer': serialize(self.next_layer)
        }
        base_config = super(WeightNormalizer, self).get_config()
        return {**base_config, **config}


object_scope = {
    'SpectralNormalization': SpectralNormalization,
    'WscaleNormalizer': WscaleNormalizer,
    'Activation': tfkl.Activation
}

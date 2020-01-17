import tensorflow as tf
import tensorflow.keras.layers as tfkl

from dynastes.layers.base_layers import DynastesBaseLayer


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class MultiHeadAttentionLayer(DynastesBaseLayer):

    def __init__(self,
                 q_layer: tfkl.Layer,
                 k_layer: tfkl.Layer,
                 v_layer: tfkl.Layer,
                 attention_layer: tfkl.Layer, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.q_layer = q_layer
        self.k_layer = k_layer
        self.v_layer = v_layer
        self.attention_layer = attention_layer

    def call(self, inputs, training=None, mask=None):

        if type(inputs) == list:
            if len(inputs) == 3:
                q, k, v = inputs
            else:
                raise ValueError('Incorrect inputs')
        else:
            raise ValueError('Incorrect inputs')

        if mask is not None:
            assert type(mask) == type(inputs)
            if type(mask) == list:
                assert len(mask) == 3
            elif type(mask) == dict:
                mask = [mask['q'], mask['k'], mask['v']]
            else:
                assert False

        q = self.q_layer(q, training=training)
        k = self.k_layer(k, training=training)
        v = self.v_layer(v, training=training)

        return self.attention_layer([q, k, v], mask=mask, training=training)

    def compute_output_shape(self, input_shape):
        assert type(input_shape) == list
        assert len(input_shape) == 3
        qs, ks, vs = input_shape
        qs = self.q_layer.compute_output_shape(qs)
        ks = self.k_layer.compute_output_shape(ks)
        vs = self.v_layer.compute_output_shape(vs)
        return self.attention_layer.compute_output_shape([qs, ks, vs])


class MultiHeadSelfAttentionLayer(DynastesBaseLayer):

    def __init__(self,
                 q_layer,
                 k_layer,
                 v_layer,
                 attention_layer, **kwargs):
        super(MultiHeadSelfAttentionLayer, self).__init__(**kwargs)
        self.multiheadAttentionLayer = MultiHeadAttentionLayer(q_layer=q_layer,
                                                               k_layer=k_layer,
                                                               v_layer=v_layer,
                                                               attention_layer=attention_layer,
                                                               name='MultiHeadAttention')

    def call(self, inputs, mask=None, training=None):
        assert type(inputs) != list
        inputs = [inputs, inputs, inputs]
        if mask is not None:
            assert type(mask) != list
            mask = [mask, mask, mask]
        return self.multiheadAttentionLayer(inputs=inputs, mask=mask, training=training)

    def compute_output_shape(self, input_shape):
        return self.multiheadAttentionLayer.compute_output_shape([input_shape, input_shape, input_shape])


class MultiHeadCrossAttentionLayer(DynastesBaseLayer):

    def __init__(self,
                 q_layer,
                 k_layer,
                 v_layer,
                 attention_layer, **kwargs):
        super(MultiHeadCrossAttentionLayer, self).__init__(**kwargs)
        self.multiheadAttentionLayer = MultiHeadAttentionLayer(q_layer=q_layer,
                                                               k_layer=k_layer,
                                                               v_layer=v_layer,
                                                               attention_layer=attention_layer,
                                                               name='MultiHeadAttention')

    def call(self, inputs, mask=None, training=None):
        assert len(inputs) == 2
        inputs = [inputs[0], inputs[1], inputs[1]]
        if mask is not None:
            mask = [mask[0], mask[1], mask[1]]
        return self.multiheadAttentionLayer(inputs=inputs, mask=mask, training=training)

    def compute_output_shape(self, input_shape):
        return self.multiheadAttentionLayer.compute_output_shape([input_shape[0], input_shape[1], input_shape[1]])

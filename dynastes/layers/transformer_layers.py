from .base_layers import DynastesBaseLayer


class MultiHeadAttentionLayer(DynastesBaseLayer):

    def __init__(self,
                 q_layer,
                 k_layer,
                 v_layer,
                 attention_layer, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.q_layer = q_layer
        self.k_layer = k_layer
        self.v_layer = v_layer
        self.attention_layer = attention_layer

    def call(self, inputs, k, v, training, mask=None, **kwargs):

        if type(inputs) == list:
            if len(inputs) == 3:
                q, k, v = inputs
            else:
                raise SyntaxError
        elif type(inputs) == dict:
            q = inputs['q']
            k = inputs['k']
            v = inputs['v']
        else:
            q = inputs

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

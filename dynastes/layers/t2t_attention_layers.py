from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dynastes.ops import t2t_attention
from dynastes.util.precision_util import large_compatible_negative
from .base_layers import DynastesBaseLayer


class LshGatingLayer(DynastesBaseLayer):

    def __init__(self,
                 bucket_length,
                 trainable=False,
                 **kwargs):
        super(LshGatingLayer, self).__init__(trainable=trainable, **kwargs)
        self.bucket_length = bucket_length
        self.buckets = 2 ** bucket_length

    def _idx_to_bits(self, i):
        """Convert an group index to its bit representation."""
        bits = bin(i)[2:].zfill(self.bucket_length)  # Pad the bits str with 0
        return [-1.0 if b == "0" else 1.0 for b in bits]

    def build(self, input_shape):
        self.t_vectors = self.add_weight(name='vector', shape=(input_shape[-1], self.bucket_length), dtype=self.dtype,
                                         initializer=tf.keras.initializers.RandomNormal(),
                                         trainable=self.trainable)
        self.t_group = tf.constant(
            [self._idx_to_bits(i) for i in range(self.buckets)],
            dtype=self.dtype,
            name="group")

    def call(self, inputs, training=None):
        # The balance loss don't propagate to the rest of the network
        x = tf.stop_gradient(inputs)
        # [length, depth] * [depth, nb_vectors * replicat]
        x = tf.matmul(x, self.get_weight('vector', training=training))
        # [length, nb_vector * replicat]
        x = tf.sign(x)  # Get on which side of the hyperplane the keys are.

        # x = tf.reshape(x, [-1, nb_replicat, nb_vector])
        # [length, replicat, nb_vector] * [nb_vector, 2^nb_vector - 1]

        x = tf.matmul(x, self.t_group, transpose_b=True) / self.bucket_length
        # We get a similarity score for each of the group between [-1, 1]
        # [length, (replicat,) 2^nb_vector - 1]
        # Do an argmax to get the most likely group for each replicat
        x = tf.argmax(x, axis=-1)
        # [length(, replicat)]
        # One-hot for compatibility with the sparse dispatcher
        x = tf.one_hot(x, self.buckets+1)
        # TODO(epot): Use a loss to force an even distribution
        return x

    def get_config(self):
        config = {'bucket_length': self.bucket_length}
        base_config = super(LshGatingLayer, self).get_config()
        return {**base_config, **config}


def _get_attention_type(local, masked, relative, self_attention, sparse):
    if self_attention:
        if sparse:
            if local:
                raise NotImplementedError
            elif relative:
                raise NotImplementedError
            else:
                attention_type = 'sparse_attention_truncated'

        else:
            if local:
                if masked:
                    attention_type = 'masked_local_attention_1d'
                else:
                    attention_type = 'unmasked_local_attention_1d'
            else:
                if relative:
                    if masked:
                        attention_type = 'masked_self_attention_relative'
                    else:
                        attention_type = 'unmasked_self_attention_relative'
                else:
                    attention_type = 'attention'
    else:
        if sparse:
            raise NotImplementedError
        if local:
            if masked:
                attention_type = 'masked_local_attention_1d'
            else:
                attention_type = 'unmasked_local_attention_1d'
        else:
            if relative:
                raise NotImplementedError
            else:
                attention_type = 'attention'
    return attention_type


class Attention1D(DynastesBaseLayer):

    def __init__(self,
                 num_heads,
                 multiquery_attention=False,
                 self_attention=False,
                 masked=False,
                 local=False,
                 relative=False,
                 sparse=False,
                 dropout_rate=0.0,
                 max_relative_position=None,
                 lsh_bucket_length=4,
                 block_length=128,
                 filter_width=100,
                 mask_right=False,
                 add_relative_to_values=False,
                 **kwargs):
        super(Attention1D, self).__init__(**kwargs)

        attention_type = _get_attention_type(local, masked, relative, self_attention, sparse)

        assert (attention_type in [
            'attention',
            'unmasked_self_attention_relative',
            'masked_self_attention_relative',
            'unmasked_local_attention_1d',
            'masked_local_attention_1d',
            'sparse_attention_truncated'])
        self.num_heads = num_heads
        self.multiquery_attention = multiquery_attention
        if self.multiquery_attention:
            self.num_heads_kv = 1
        else:
            self.num_heads_kv = num_heads
        self.attention_type = attention_type
        self.masked = masked
        self.local = local
        self.relative = relative
        self.sparse = sparse
        self.mask_right = mask_right
        self.dropout_rate = dropout_rate
        self.lsh_bucket_length = lsh_bucket_length
        self.max_relative_position = max_relative_position
        self.block_length = block_length
        self.filter_width = filter_width
        self.lsh_gates = None
        self.add_relative_to_values = add_relative_to_values
        self.maybe_build_lsh_gates()

    def maybe_build_lsh_gates(self):
        if self.attention_type == 'sparse_attention_truncated' and self.num_heads is not None and self.lsh_gates is None:
            self.lsh_gates = [LshGatingLayer(bucket_length=self.lsh_bucket_length) for _ in range(self.num_heads)]

    def build(self, input_shape):
        self.depth_q = int(input_shape[0][-1]) // self.num_heads
        depth_k = int(input_shape[1][-1]) // self.num_heads_kv
        depth_v = int(input_shape[2][-1]) // self.num_heads_kv
        if 'relative' in self.attention_type:
            if self.attention_type == 'unmasked_self_attention_relative':
                k_embedding_shape = t2t_attention.get_relative_embeddings_left_right_shape(
                    self.max_relative_position,
                    depth_k, self.num_heads_kv,
                    False)
                self.key_embeddings = self.add_weight('key_embeddings',
                                                      initializer=tf.keras.initializers.RandomNormal(
                                                          stddev=t2t_attention.get_embedding_initializer_stddev(
                                                              depth_k)),
                                                      shape=k_embedding_shape)

                if self.add_relative_to_values:
                    v_embedding_shape = t2t_attention.get_relative_embeddings_left_right_shape(
                        self.max_relative_position,
                        depth_v, self.num_heads_kv,
                        False)
                    self.value_embeddings = self.add_weight('value_embeddings',
                                                            initializer=tf.keras.initializers.RandomNormal(
                                                                stddev=t2t_attention.get_embedding_initializer_stddev(
                                                                    depth_v)),
                                                            shape=v_embedding_shape)
                else:
                    self.value_embeddings = None
            elif self.attention_type == 'masked_self_attention_relative':
                k_embedding_shape = t2t_attention.get_relative_embeddings_left_shape(
                    self.max_relative_position,
                    depth_k, self.num_heads_kv, False)
                self.key_embeddings = self.add_weight('key_embeddings',
                                                      initializer=tf.keras.initializers.RandomNormal(
                                                          stddev=t2t_attention.get_embedding_initializer_stddev(
                                                              depth_k)),
                                                      shape=k_embedding_shape)
                if self.add_relative_to_values:
                    v_embedding_shape = t2t_attention.get_relative_embeddings_left_shape(
                        self.max_relative_position,
                        depth_v, self.num_heads_kv,
                        False)
                    self.value_embeddings = self.add_weight('value_embeddings',
                                                            initializer=tf.keras.initializers.RandomNormal(
                                                                stddev=t2t_attention.get_embedding_initializer_stddev(
                                                                    depth_v)),
                                                            shape=v_embedding_shape)
                else:
                    self.value_embeddings = None
            else:
                raise ValueError()

        self.maybe_build_lsh_gates()
        super(Attention1D, self).build(input_shape)

    def _create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def call(self, inputs, training=None, mask=None):
        if len(inputs) == 3:
            q, k, v = inputs
        else:
            raise ValueError()

        if mask is not None:
            q_mask = (1. - tf.cast(mask[0], tf.float32))[:, tf.newaxis, :, tf.newaxis]
            kv_mask = (1. - tf.cast(mask[1], tf.float32))[:, tf.newaxis, tf.newaxis, :]
            c_mask = tf.maximum(q_mask, kv_mask)
            if self.mask_right:
                look_ahead_mask = self._create_look_ahead_mask(mask[0].shape[1])
                c_mask = tf.maximum(c_mask, look_ahead_mask)
            bias = c_mask * large_compatible_negative(k.dtype)

        r = None
        weights = None

        q = t2t_attention.split_heads(q, self.num_heads)
        k = t2t_attention.split_heads(k, self.num_heads_kv)
        v = t2t_attention.split_heads(v, self.num_heads_kv)

        if 'relative' in self.attention_type:
            key_embeddings = self.get_weight('key_embeddings', training=training)
            if self.add_relative_to_values:
                value_embeddings = self.get_weight('value_embeddings', training=training)
            else:
                value_embeddings = None
            if self.attention_type == 'unmasked_self_attention_relative':
                r, weights = t2t_attention.dot_product_unmasked_self_attention_relative_v2(q=q, k=k, v=v, bias=bias,
                                                                                           key_leftright_embeddings=key_embeddings,
                                                                                           value_leftright_embeddings=value_embeddings,
                                                                                           dropout_rate=self.dropout_rate,
                                                                                           max_relative_position=self.max_relative_position)
            elif self.attention_type == 'masked_self_attention_relative':
                r, weights = t2t_attention.dot_product_self_attention_relative_v2(q=q, k=k, v=v, bias=bias,
                                                                                  key_left_embedding=key_embeddings,
                                                                                  value_left_embedding=value_embeddings,
                                                                                  dropout_rate=self.dropout_rate,
                                                                                  max_relative_position=self.max_relative_position)
        else:
            if self.attention_type == 'unmasked_local_attention_1d':
                r, weights = t2t_attention.local_attention_1d(q=q, k=k, v=v, block_length=self.block_length,
                                                              filter_width=self.filter_width)
            elif self.attention_type == 'masked_local_attention_1d':
                r, weights = t2t_attention.masked_local_attention_1d(q=q, k=k, v=v, block_length=self.block_length,
                                                                     dropout_rate=self.dropout_rate)
            elif self.attention_type == 'sparse_attention_truncated':
                r, loss, weights = t2t_attention.sparse_dot_product_attention_truncated(q=q, k=k, v=v,
                                                                                        list_lsh=self.lsh_gates,
                                                                                        mask_right=self.mask_right)
                self.add_loss(loss)
            else:
                r, weights = t2t_attention.dot_product_attention(q=q, k=k, v=v, bias=bias,
                                                                 dropout_rate=self.dropout_rate)

        r = t2t_attention.combine_heads(r)
        return r, weights

    def get_config(self):
        config = {
            'num_heads_q': self.num_heads_q,
            'masked': self.masked,
            'local': self.local,
            'relative': self.relative,
            'sparse': self.sparse,
            'dropout_rate': self.dropout_rate,
            'max_relative_position': self.max_relative_position,
            'lsh_bucket_length': self.lsh_bucket_length,
            'block_length': self.block_length,
            'filter_width': self.filter_width,
            'mask_right': self.mask_right,
            'add_relative_to_values': self.add_relative_to_values,
        }
        base_config = super(Attention1D, self).get_config()
        return {**base_config, **config}

import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.sparse_ops import sparse_split_v2

from .grad_helpers import custom_gradient_back_fn


def sparse_dense_matmul(a, b):
    def _sparse_dense_matmul_deep_support(st, dt, st_shape=None, dt_shape=None):
        if st_shape is None:
            st_shape = st.get_shape().as_list()
        if len(st_shape) == 2:
            if len(dt_shape) > 2:
                return _sparse_dense_matmul_deep_support(st, dt[0], st_shape=st_shape, dt_shape=dt_shape[1:])
            return tf.sparse.sparse_dense_matmul(st, dt)
        if dt_shape is None:
            dt_shape = dt.get_shape().as_list()
        st_split = sparse_split_v2(st, num_split=st_shape[0], axis=0)
        res = []
        n_shape = st_shape[1:]
        n_dt_shape = dt_shape[1:]
        if st_shape[0] != dt_shape[0]:
            for i in range(st_shape[0]):
                ss = st_split[i]
                ss = tf.sparse.reshape(ss, shape=n_shape)
                r = _sparse_dense_matmul_deep_support(ss, dt, st_shape=n_shape, dt_shape=dt_shape)
                res.append(tf.expand_dims(r, axis=0))
        else:
            dt_split = tf.split(dt, num_or_size_splits=dt_shape[0], axis=0)
            for i in range(st_shape[0]):
                ss = st_split[i]
                ds = dt_split[i]
                ss = tf.sparse.reshape(ss, shape=n_shape)
                ds = tf.reshape(ds, shape=n_dt_shape)
                r = _sparse_dense_matmul_deep_support(ss, ds, st_shape=n_shape, dt_shape=n_dt_shape)
                res.append(tf.expand_dims(r, axis=0))
        return tf.concat(res, axis=0)

    return _sparse_dense_matmul_deep_support(a, b, a.get_shape().as_list(), b.get_shape().as_list())


def bit_population_count(x):
    return bitwise_ops.PopulationCount(x=x)


def _idx_to_bits(i, bucket_length):
    """Convert an group index to its bit representation."""
    bits = bin(i)[2:].zfill(bucket_length)  # Pad the bits str with 0
    return [-1.0 if b == "0" else 1.0 for b in bits]


def _get_lsh_clusters(x, t_group, bucket_length, threshold=0.5, dtype=tf.float16):
    x = tf.sign(x)  # Get on which side of the hyperplane the keys are.

    # Prefer native implementation
    if bucket_length % 8 == 0:
        return gen_math_ops.compare_and_bitpack(input=x, threshold=threshold)
    # x = tf.reshape(x, [-1, nb_replicat, nb_vector])
    # [length, replicat, nb_vector] * [nb_vector, 2^nb_vector - 1]

    x = tf.matmul(x, tf.cast(t_group, dtype), transpose_b=True) / bucket_length
    x = tf.expand_dims(x, axis=-1)
    # We get a similarity score for each of the group between [-1, 1]
    # [length, (replicat,) 2^nb_vector - 1]
    # Do an argmax to get the most likely group for each replica
    x = tf.argmax(x, axis=-1)
    return x


def _mul_lsh_vectors(x, t_vectors, dtype=tf.float16):
    y = tf.stop_gradient(x)
    y = tf.cast(y, dtype)
    # [length, depth] * [depth, nb_vectors * replicat]
    return tf.matmul(y, t_vectors)


def _lsh_similarity_grad_forward(x, y, t_vectors_x, t_group, bucket_length, threshold=0.5,
                                 dtype=tf.float16,
                                 t_vectors_y=None):
    t_vectors_x = tf.cast(t_vectors_x, dtype)
    if t_vectors_y is None:
        t_vectors_y = t_vectors_x
    else:
        t_vectors_y = tf.cast(t_vectors_y, dtype)

    xh = tf.matmul(tf.cast(x, dtype), t_vectors_x)
    yh = tf.matmul(tf.cast(y, dtype), t_vectors_y)

    xb = tf.matmul(xh, t_group, transpose_b=True) / bucket_length
    yb = tf.matmul(yh, t_group, transpose_b=True) / bucket_length

    # xb_norm = nn.l2_normalize(xb, axis=-1)
    # yb_norm = nn.l2_normalize(yb, axis=-1)

    # xb_norm = tf.expand_dims(xb_norm, axis=-1)
    # similarity = math_ops.reduce_sum(xb_norm * yb_norm, axis=-1)

    return nn.l2_normalize(tf.matmul(xb, yb, transpose_b=True))


def _lsh_similarity(x, y, t_vectors_x, t_group, bucket_length, threshold=0.5,
                    dtype=tf.float16,
                    t_vectors_y=None):
    t_vectors_x = tf.cast(t_vectors_x, dtype)
    if t_vectors_y is None:
        t_vectors_y = t_vectors_x
    else:
        t_vectors_y = tf.cast(t_vectors_y, dtype)
    xh = _mul_lsh_vectors(x, t_vectors_x, dtype=dtype)
    yh = _mul_lsh_vectors(y, t_vectors_y, dtype=dtype)

    xb = _get_lsh_clusters(xh, t_group=t_group, bucket_length=bucket_length)
    yb = _get_lsh_clusters(yh, t_group=t_group, bucket_length=bucket_length)

    bw_sim = tf.bitwise.bitwise_xor(tf.expand_dims(xb, axis=-3), tf.expand_dims(yb, axis=-2))
    p_count = bit_population_count(bw_sim)
    p_count = tf.reduce_sum(p_count, axis=-1)
    indices = tf.where(p_count < int(bucket_length * threshold))
    distances = tf.cast((bucket_length - tf.gather_nd(p_count, indices)), dtype) / bucket_length
    return tf.SparseTensor(indices, distances, dense_shape=p_count.shape)


def lsh_similary_back_fn(x, y, t_vectors, t_group, bucket_length, threshold=0.5,
                         dtype=tf.float16,
                         t_vectors_y=None):
    return _lsh_similarity_grad_forward(x, y,
                                        t_vectors_x=t_vectors,
                                        t_group=t_group,
                                        bucket_length=bucket_length,
                                        threshold=threshold,
                                        dtype=dtype,
                                        t_vectors_y=t_vectors_y)


@custom_gradient_back_fn(lsh_similary_back_fn)
def lsh_similarity(x, y, t_vectors, t_group, bucket_length, threshold=0.5,
                   dtype=tf.float16,
                   t_vectors_y=None):
    r = _lsh_similarity(x, y,
                        t_vectors_x=t_vectors,
                        t_group=t_group,
                        bucket_length=bucket_length,
                        threshold=threshold,
                        dtype=dtype,
                        t_vectors_y=t_vectors_y)
    return tf.sparse.to_dense(r, default_value=0.)


def lsh_attention_back_fn(q, k, v, t_vectors_q, t_group, bucket_length, threshold=0.5, dtype=tf.float16,
                          t_vectors_k=None):
    logits = _lsh_similarity_grad_forward(q, k,
                                          t_vectors_x=t_vectors_q,
                                          t_group=t_group,
                                          bucket_length=bucket_length,
                                          threshold=threshold,
                                          dtype=dtype,
                                          t_vectors_y=t_vectors_k)
    attention_weights = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)
    return tf.matmul(attention_weights, v)

##
# WARNING: This is slow, really just an experiment, do not use
##
@custom_gradient_back_fn(lsh_attention_back_fn)
def lsh_attention(q, k, v, t_vectors_q, t_group, bucket_length, threshold=0.5, dtype=tf.float16, t_vectors_k=None):
    logits = _lsh_similarity(q, k,
                             t_vectors_x=t_vectors_q,
                             t_group=t_group,
                             bucket_length=bucket_length,
                             threshold=threshold,
                             dtype=dtype,
                             t_vectors_y=t_vectors_k)
    attention_weights = tf.sparse.softmax(tf.cast(logits, tf.float32))
    if tf.executing_eagerly:
        return sparse_dense_matmul(attention_weights, v)
    else:
        return tf.matmul(tf.sparse.to_dense(attention_weights), v)
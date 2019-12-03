import abc

import tensorflow as tf

from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.ops.t2t_common import shape_list
from dynastes.probability import bijector_partials
from dynastes.probability.bijectors import EventShapeAwareChain


class PseudoBlockSparseBijector(DynastesBaseLayer, abc.ABC):

    @abc.abstractmethod
    def get_bijector(self, x):
        pass

    @abc.abstractmethod
    def get_causality_matrix(self, x):
        pass


class PseudoBlockSparseBijector1D(PseudoBlockSparseBijector, abc.ABC):

    def get_causality_matrix(self, x):
        _, length, _ = shape_list(x)
        lrange = tf.range(0, length)
        return tf.reshape(lrange, [1, length, 1])


class ChainedPseudoBlockSparseBijector1D(PseudoBlockSparseBijector1D):

    def __init__(self, partial_bijectors, **kwargs):
        super().__init__(**kwargs)
        self.partial_bijectors = partial_bijectors

    def get_bijector(self, x):
        event_shape_in = shape_list(x)[1:]
        chain = EventShapeAwareChain(event_shape_in, self.partial_bijectors)
        return chain

    def get_causality_matrix(self, x):
        lrange = super(ChainedPseudoBlockSparseBijector1D, self).get_causality_matrix(x)
        return self.get_bijector(lrange).forward(lrange)


class BlockSparseStridedRoll1D(ChainedPseudoBlockSparseBijector1D):

    def __init__(self,
                 block_size,
                 stride=1,
                 partial_bijectors=None,
                 **kwargs):
        partial_bijectors = [
            bijector_partials.Reshape([-1, block_size, -1]),
            bijector_partials.RollIncremental(axis=1, steps=-(block_size + stride), constant=0),
            bijector_partials.Transpose(perm=[1, 0, 2]),
            # bijector_partials.RollIncremental(axis=1, steps=-2, constant=0),
            bijector_partials.Reshape([-1, -1])
        ]
        super(BlockSparseStridedRoll1D, self).__init__(partial_bijectors=partial_bijectors, **kwargs)

class BlockSparsePrimedStridedRoll1D(ChainedPseudoBlockSparseBijector1D):

    def __init__(self,
                 block_size,
                 n_primes=-1,
                 stride=1,
                 partial_bijectors=None,
                 **kwargs):

        def prime_factors(n):
            i = 2
            factors = []
            while i * i <= n:
                if n % i:
                    i += 1
                else:
                    n //= i
                    factors.append(i)
            if n > 1:
                factors.append(n)
            return factors

        primes = prime_factors(block_size)
        if n_primes != -1:
            primes = primes[:n_primes-1] + [sum(primes[n_primes-1:])]
        else:
            n_primes = len(primes)


        partial_bijectors = [
            bijector_partials.Reshape([-1] + primes + [-1])
        ]
        for i in range(n_primes):
            partial_bijectors.append(
                bijector_partials.RollIncremental(axis=1+i, steps=-(primes[i] + stride), constant=0)
            )
        partial_bijectors.append(
            bijector_partials.Transpose(perm=[0] + list(reversed(range(1, n_primes))) + [n_primes])
        )
        partial_bijectors.append(
            bijector_partials.Reshape([-1, -1])
        )
        super(BlockSparsePrimedStridedRoll1D, self).__init__(partial_bijectors=partial_bijectors, **kwargs)


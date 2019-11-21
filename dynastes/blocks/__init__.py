from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .attention_blocks import SelfAttentionBlock1D

object_scope = {
    'SelfAttentionBlock1D': SelfAttentionBlock1D,
}

# Cleanup symbols to avoid polluting namespace.
del absolute_import
del division
del print_function

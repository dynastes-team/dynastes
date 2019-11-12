from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .localized_attention_blocks import LocalizedSelfAttentionBlock1D

object_scope = {
    'LocalizedSelfAttentionBlock1D': LocalizedSelfAttentionBlock1D,
}

# Cleanup symbols to avoid polluting namespace.
del absolute_import
del division
del print_function

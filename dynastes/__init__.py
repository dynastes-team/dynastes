from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dynastes import blocks
from dynastes import core
from dynastes import helpers
from dynastes import layers
from dynastes import models
from dynastes import ops
from dynastes import probability
from dynastes import regularizers
from dynastes import util
from dynastes import weight_normalizers
from dynastes.core import backend
from dynastes.probability import bijectors, bijector_partials, pseudoblocksparse_bijectors

object_scope = {
    **blocks.object_scope,
    **layers.object_scope,
    **weight_normalizers.object_scope,
    **regularizers.object_scope,
}

__version__ = "0.3.8"
# Cleanup symbols to avoid polluting namespace.
del absolute_import
del division
del print_function

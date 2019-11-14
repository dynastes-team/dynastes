from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.bitwise_ops import PopulationCount
from dynastes import blocks
from dynastes import layers
from dynastes import normalizers
from dynastes import ops
from dynastes import regularizers
from dynastes import util
from dynastes.core import backend
from dynastes import helpers

object_scope = {
    **blocks.object_scope,
    **layers.object_scope,
    **normalizers.object_scope,
    **regularizers.object_scope,
}

__version__ = "0.1.9"
# Cleanup symbols to avoid polluting namespace.
del absolute_import
del division
del print_function

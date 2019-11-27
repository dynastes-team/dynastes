from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dynastes import blocks
from dynastes import core
from dynastes import helpers
from dynastes import layers
from dynastes import models
from dynastes import normalizers
from dynastes import ops
from dynastes import regularizers
from dynastes import util
from dynastes.core import backend

object_scope = {
    **blocks.object_scope,
    **layers.object_scope,
    **normalizers.object_scope,
    **regularizers.object_scope,
}

__version__ = "0.2.2"
# Cleanup symbols to avoid polluting namespace.
del absolute_import
del division
del print_function

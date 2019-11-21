from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .attention_layers import LocalizedAttentionLayer1D
from .attention_layers import LocalizedAttentionLayer2D
from .t2t_attention_layers import Attention1D
from .base_layers import ActivatedKernelBiasBaseLayer
from .time_delay_layers import DepthGroupwiseTimeDelayLayer1D
from .time_delay_layers import DepthGroupwiseTimeDelayLayerFake2D
from .time_delay_layers import TimeDelayLayer1D
from .time_delay_layers import TimeDelayLayerFake2D

object_scope = {
    'LocalizedAttentionLayer1D': LocalizedAttentionLayer1D,
    'LocalizedAttentionLayer2D': LocalizedAttentionLayer2D,
    'ActivatedKernelBiasBaseLayer': ActivatedKernelBiasBaseLayer,
    'DepthGroupwiseTimeDelayLayer1D': DepthGroupwiseTimeDelayLayer1D,
    'DepthGroupwiseTimeDelayLayerFake2D': DepthGroupwiseTimeDelayLayerFake2D,
    'TimeDelayLayer1D': TimeDelayLayer1D,
    'TimeDelayLayerFake2D': TimeDelayLayerFake2D,
    'Attention1D': Attention1D
}

# Cleanup symbols to avoid polluting namespace.
del absolute_import
del division
del print_function

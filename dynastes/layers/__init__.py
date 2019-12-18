from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .attention_layers import LocalizedAttentionLayer1D
from .attention_layers import LocalizedAttentionLayer2D
from .base_layers import ActivatedKernelBiasBaseLayer, _WscaleInitializer, DynastesDense
from .conditioning_layers import FeaturewiseLinearModulation
from .convolutional_layers import DynastesConv1D
from .convolutional_layers import DynastesConv1DTranspose
from .convolutional_layers import DynastesConv2D
from .convolutional_layers import DynastesConv2DTranspose
from .convolutional_layers import DynastesConv3D
from .convolutional_layers import DynastesDepthwiseConv1D
from .convolutional_layers import DynastesDepthwiseConv2D
from .normalization_layers import AdaptiveGroupNormalization
from .normalization_layers import AdaptiveInstanceNormalization
from .normalization_layers import AdaptiveLayerInstanceNormalization
from .normalization_layers import AdaptiveLayerNormalization
from .normalization_layers import AdaptiveMultiNormalization
from .normalization_layers import MultiNormalization
from .normalization_layers import PoolNormalization2D
from .random_layers import StatelessRandomNormalLike
from .t2t_attention_layers import AddTimingSignalLayer1D
from .t2t_attention_layers import Attention1D
from .t2t_attention_layers import Attention2D
from .t2t_attention_layers import LshGatingLayer
from .t2t_attention_layers import PseudoBlockSparseAttention1D
from .time_delay_layers import DepthGroupwiseTimeDelayLayer1D
from .time_delay_layers import DepthGroupwiseTimeDelayLayerFake2D
from .time_delay_layers import TimeDelayLayer1D
from .time_delay_layers import TimeDelayLayerFake2D

object_scope = {
    '_WscaleInitializer': _WscaleInitializer,
    'DynastesDense': DynastesDense,
    'LocalizedAttentionLayer1D': LocalizedAttentionLayer1D,
    'LocalizedAttentionLayer2D': LocalizedAttentionLayer2D,
    'ActivatedKernelBiasBaseLayer': ActivatedKernelBiasBaseLayer,
    'DepthGroupwiseTimeDelayLayer1D': DepthGroupwiseTimeDelayLayer1D,
    'DepthGroupwiseTimeDelayLayerFake2D': DepthGroupwiseTimeDelayLayerFake2D,
    'TimeDelayLayer1D': TimeDelayLayer1D,
    'TimeDelayLayerFake2D': TimeDelayLayerFake2D,
    'AddTimingSignalLayer1D': AddTimingSignalLayer1D,
    'LshGatingLayer': LshGatingLayer,
    'Attention1D': Attention1D,
    'Attention2D': Attention2D,
    'PseudoBlockSparseAttention1D': PseudoBlockSparseAttention1D,
    'StatelessRandomNormalLike': StatelessRandomNormalLike,
    'PoolNormalization2D': PoolNormalization2D,
    'MultiNormalization': MultiNormalization,
    'AdaptiveMultiNormalization': AdaptiveMultiNormalization,
    'AdaptiveLayerInstanceNormalization': AdaptiveLayerInstanceNormalization,
    'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization,
    'AdaptiveGroupNormalization': AdaptiveGroupNormalization,
    'AdaptiveLayerNormalization': AdaptiveLayerNormalization,
    'FeaturewiseLinearModulation': FeaturewiseLinearModulation,
    'DynastesConv1D': DynastesConv1D,
    'DynastesConv2D': DynastesConv2D,
    'DynastesConv3D': DynastesConv3D,
    'DynastesConv1DTranspose': DynastesConv1DTranspose,
    'DynastesConv2DTranspose': DynastesConv2DTranspose,
    'DynastesDepthwiseConv1D': DynastesDepthwiseConv1D,
    'DynastesDepthwiseConv2D': DynastesDepthwiseConv2D,
}

# Cleanup symbols to avoid polluting namespace.
del absolute_import
del division
del print_function

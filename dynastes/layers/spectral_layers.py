import math

import tensorflow as tf

from dynastes.layers.base_layers import DynastesBaseLayer
from dynastes.ops import spectral_ops


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class Wave2STFTLayer(DynastesBaseLayer):

    def __init__(self,
                 n_bins,
                 hop_length,
                 padding,
                 keep_dc=False,
                 hq=True,
                 **kwargs):
        super(Wave2STFTLayer, self).__init__(**kwargs)
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.padding = padding
        self.keep_dc = keep_dc
        self.hq = hq
        self.inverted = False

    def call(self, inputs, invert=False, **kwargs):

        if invert or self.inverted:
            stfts = inputs
            if self.hq:
                stfts = tf.cast(stfts, tf.complex128)
            waves = spectral_ops.stfts_to_waves(stfts,
                                                n_fft=self.n_bins * 2,
                                                hop_length=self.hop_length,
                                                pad_l=self.padding,
                                                pad_r=self.padding,
                                                discard_dc=not self.keep_dc)
            return waves

        stfts = spectral_ops.waves_to_stfts(inputs,
                                            n_fft=self.n_bins * 2,
                                            hop_length=self.hop_length,
                                            pad_l=self.padding,
                                            pad_r=self.padding,
                                            discard_dc=not self.keep_dc,
                                            hq=self.hq)
        return stfts#tf.cast(stfts, inputs.dtype)

    def compute_output_shape(self, input_shape, invert=False):
        if invert:
            self.inverted = True
            output_shape = super(Wave2STFTLayer, self).compute_output_shape(input_shape)
            self.inverted = False
            return output_shape
        out_bins = self.n_bins
        if self.keep_dc:
            out_bins += 1
        print(input_shape)
        out_len = max(0, math.floor(((input_shape[1] + (self.padding * 2) - (self.n_bins * 2)) / self.hop_length)) + 1)
        output_shape = [input_shape[0], out_len, out_bins, input_shape[-1]]
        print(output_shape)
        return tf.TensorShape(dims=output_shape)

    def compute_output_signature(self, input_signature: tf.TensorSpec, invert=False):
        if invert:
            self.inverted = True
            output_signature = super(Wave2STFTLayer, self).compute_output_signature(input_signature)
            self.inverted = False
            return output_signature
        else:
            out_bins = self.n_bins
            if self.keep_dc:
                out_bins += 1
            output_dtype = tf.complex64
            if self.hq:
                output_dtype = tf.complex128
            input_shape = input_signature.shape.as_list()
            out_len = max(0, math.floor(((input_shape[1] + (self.padding * 2) - (self.n_bins * 2)) / self.hop_length)) + 1)
            output_shape = tf.TensorShape([input_shape[0], out_len, out_bins, input_shape[2]])
            return tf.TensorSpec(shape=output_shape, dtype=output_dtype)

    def get_config(self):
        config = {'n_bins': self.n_bins,
                  'hop_length': self.hop_length,
                  'padding': self.padding,
                  'keep_dc': self.keep_dc,
                  'hq': self.hq}
        base_config = super(Wave2STFTLayer, self).get_config()
        return {**base_config, **config}


def _get_lower_and_upper_edges_l2mel(min_hz, max_hz, n_mels):
    mxhzl2 = math.log2(max_hz)
    mnhzl2 = math.log2(min_hz)

    l2pmel = (mxhzl2 - mnhzl2) / n_mels

    low_edge_l2 = mnhzl2 - (l2pmel / 2)
    high_edge_l2 = mxhzl2 + (l2pmel / 2)

    low_edge_hz = 2 ** low_edge_l2
    high_edge_hz = 2 ** high_edge_l2
    return low_edge_hz, high_edge_hz


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class _l2melMatrixInitializer(tf.keras.initializers.Initializer):

    def __init__(self,
                 min_hz=40.,
                 max_hz=22050.,
                 sr=44100.,
                 smooth_l2mel=False):
        super(_l2melMatrixInitializer, self).__init__()
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.sr = sr
        self.smooth_l2mel = smooth_l2mel

    def __call__(self, shape,
                 dtype=None):
        if dtype is None:
            dtype = tf.float32
        lower_edge_hz, higher_edge_hz = _get_lower_and_upper_edges_l2mel(self.min_hz, self.max_hz, shape[1])
        num_spectrogram_bins = shape[0]
        if self.smooth_l2mel:
            num_spectrogram_bins = min(shape(0) * 4, shape[1] * 8)

        l2melmat = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=shape[1],
                                                         num_spectrogram_bins=num_spectrogram_bins,
                                                         lower_edge_hertz=lower_edge_hz,
                                                         upper_edge_hertz=higher_edge_hz,
                                                         sample_rate=self.sr,
                                                         dtype=dtype)
        if self.smooth_l2mel:
            l2melmat = tf.expand_dims(l2melmat, axis=-1)
            l2melmat = tf.image.resize(l2melmat, shape, method='mitchellcubic', antialias=True)
            l2melmat = tf.squeeze(l2melmat, axis=-1)

        return l2melmat

    def get_config(self):
        config = {'min_hz': self.min_hz,
                  'max_hz': self.max_hz,
                  'sr': self.sr,
                  'smooth_l2mel': self.smooth_l2mel}
        base_config = super(_l2melMatrixInitializer, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class STFT2MelspectrogramLayer(DynastesBaseLayer):

    def __init__(self,
                 n_mels,
                 min_hz=40.,
                 max_hz=20000.,
                 sr=44100.,
                 trainable=False,
                 smooth_l2mel=False,
                 ifreq=False,
                 return_phase=False,
                 **kwargs):
        super(STFT2MelspectrogramLayer, self).__init__(trainable=trainable, **kwargs)
        self.n_mels = n_mels
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.sr = sr
        self.smooth_l2mel = smooth_l2mel
        self.ifreq = ifreq
        self.return_phase = return_phase
        self.inverted = False

    def build(self, input_shape):
        n_bins = input_shape[-2]
        l2mel_shape = [n_bins, self.n_mels]
        self.add_weight('l2mel', shape=l2mel_shape,
                        trainable=self.trainable,
                        initializer=_l2melMatrixInitializer(min_hz=self.min_hz,
                                                            max_hz=self.max_hz,
                                                            sr=self.sr,
                                                            smooth_l2mel=self.smooth_l2mel))

    def call(self, inputs, training=None, invert=False, **kwargs):
        if invert or self.inverted:
            stfts = spectral_ops.melspecgrams_to_stfts(melspecgrams=inputs,
                                                       mel2l=tf.transpose(self.get_weight('l2mel', training=training)),
                                                       ifreq=self.ifreq and self.return_phase)
            return stfts
        mel_spectrograms = spectral_ops.stfts_to_melspecgrams(stfts=inputs,
                                                              l2mel=self.get_weight('l2mel', training=training),
                                                              ifreq=self.ifreq,
                                                              return_phase=self.return_phase)
        mel_spectrograms = tf.cast(mel_spectrograms, self.dtype)
        return mel_spectrograms

    def get_config(self):
        config = {'n_mels': self.n_mels,
                  'min_hz': self.min_hz,
                  'max_hz': self.max_hz,
                  'sr': self.sr,
                  'smooth_l2mel': self.smooth_l2mel,
                  'ifreq': self.ifreq,
                  'return_phase': self.return_phase}
        base_config = super(STFT2MelspectrogramLayer, self).get_config()
        return {**base_config, **config}

    def compute_output_signature(self, input_signature, invert=False):
        if invert:
            self.inverted = True
            output_signature = super(STFT2MelspectrogramLayer, self).compute_output_signature(input_signature)
            self.inverted = False
            return output_signature
        else:
            input_shape = input_signature.shape.as_list()
            n_channels = input_shape[3]
            if self.return_phase:
                n_channels *= 2
            output_shape = tf.TensorShape([input_shape[0], input_shape[1], self.n_mels, n_channels])
            return tf.TensorSpec(shape=output_shape, dtype=self.dtype)

    def compute_output_shape(self, input_shape, invert=False):
        if invert:
            self.inverted = True
            output_shape = super(STFT2MelspectrogramLayer, self).compute_output_shape(input_shape)
            self.inverted = False
            return output_shape
        n_channels = input_shape[3]
        if self.return_phase:
            n_channels *= 2
        return tf.TensorShape([input_shape[0], input_shape[1], self.n_mels, n_channels])

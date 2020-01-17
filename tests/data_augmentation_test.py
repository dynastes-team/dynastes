import os

import tensorflow as tf
from tensorflow.python.framework import test_util

from dynastes.data_augmentation.audio_spectrum import spec_augment
from dynastes.ops import spectral_ops


class SpecAugmentTest(tf.test.TestCase):

    @test_util.use_deterministic_cudnn
    def test_simple(self):
        n_fft = 256
        hop_length = 64
        pad = 256
        dir_path = os.path.dirname(os.path.realpath(__file__))

        raw_audio = tf.io.read_file(dir_path + '/data/test_wav.wav')
        waves = tf.audio.decode_wav(raw_audio)[0][:20480] * 0.99
        waves.shape
        waves = tf.expand_dims(waves, 0)

        print(waves.shape)

        @tf.function
        def graph_test(waves):
            stfts = spectral_ops.waves_to_stfts(waves, n_fft=n_fft, hop_length=hop_length, pad_l=pad, pad_r=pad,
                                                discard_dc=True, hq=False)
            return stfts
        stfts = graph_test(waves)


        stfts_mags = tf.maximum(tf.minimum(tf.abs(stfts), 0.5), -0.5)
        stfts_angles = tf.math.angle(stfts)
        stfts = spectral_ops.polar2rect(stfts_mags, stfts_angles)

        l2mel = tf.signal.linear_to_mel_weight_matrix(num_spectrogram_bins=n_fft // 2,
                                                      sample_rate=44100,
                                                      lower_edge_hertz=50.,
                                                      upper_edge_hertz=6000.,
                                                      num_mel_bins=80)

        melmags = spectral_ops.stfts_to_melspecgrams(stfts, l2mel=l2mel, ifreq=False, return_phase=False)
        # melmags = tf.reduce_sum(melmags, axis=-1)
        print(melmags.shape)

        augs = []
        print(tf.reduce_max(melmags), tf.reduce_min(melmags))

        for i in range(50):
            augs.append(spec_augment(melmags))

        mean_augs = tf.reduce_mean(tf.stack(augs), axis=0)

        diff = tf.reduce_mean(melmags - mean_augs)
        print(diff)

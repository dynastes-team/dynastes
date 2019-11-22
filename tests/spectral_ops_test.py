import os

import tensorflow as tf
from tensorflow.python.framework import test_util

from dynastes.ops import spectral_ops


class SpectralOpsTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        n_fft = 256
        hop_length = 256
        pad = 8192 * 4
        dir_path = os.path.dirname(os.path.realpath(__file__))

        raw_audio = tf.io.read_file(dir_path + '/data/test_wav.wav')
        waves = tf.audio.decode_wav(raw_audio)[0][:20480] * 0.25
        waves = tf.expand_dims(waves, 0)
        waves = tf.expand_dims(waves, -1)
        waves = tf.keras.layers.UpSampling2D((2, 1), interpolation='bilinear')(waves)
        waves = tf.squeeze(waves, axis=-1)

        stfts = spectral_ops.waves_to_stfts(waves, n_fft=n_fft, hop_length=hop_length, pad_l=pad, pad_r=pad,
                                            discard_dc=False)
        stfts_mags = tf.maximum(tf.minimum(tf.abs(stfts), 0.5), -0.5)
        stfts_angles = tf.math.angle(stfts)
        stfts = spectral_ops.polar2rect(stfts_mags, stfts_angles)

        waves_recon = spectral_ops.stfts_to_waves(stfts, n_fft=n_fft, hop_length=hop_length, pad_l=pad, pad_r=pad,
                                                  discard_dc=False)
        assert (tf.abs(tf.reduce_mean(tf.abs(waves) - tf.abs(waves_recon))).numpy() < 0.1)
        waves_recon_stft = spectral_ops.waves_to_stfts(waves_recon, n_fft=n_fft, hop_length=hop_length, pad_l=pad,
                                                       pad_r=pad, discard_dc=False)
        assert (tf.abs(tf.reduce_mean(tf.abs(stfts) - tf.abs(waves_recon_stft))).numpy() < 0.1)
        l2mel = tf.signal.linear_to_mel_weight_matrix(num_spectrogram_bins=(n_fft // 2) + 1, num_mel_bins=96)

        # Mel transform
        melspecgrams = spectral_ops.stfts_to_melspecgrams(stfts, l2mel=l2mel)
        stfts_mel_recon = spectral_ops.melspecgrams_to_stfts(melspecgrams, mel2l=tf.transpose(l2mel))
        melspecgram_stfts_mel_recon = spectral_ops.stfts_to_melspecgrams(stfts_mel_recon, l2mel=l2mel)

        assert (tf.abs(tf.reduce_mean(tf.abs(melspecgrams) - tf.abs(melspecgram_stfts_mel_recon))).numpy() < 0.4)

        stfts_mel_recon_mel_recon = spectral_ops.melspecgrams_to_stfts(melspecgram_stfts_mel_recon,
                                                                       mel2l=tf.transpose(l2mel))
        assert (tf.abs(tf.reduce_mean(tf.abs(stfts_mel_recon) - tf.abs(stfts_mel_recon_mel_recon))).numpy() < 0.01)
        waves_mel_recon = spectral_ops.stfts_to_waves(stfts_mel_recon, n_fft=n_fft, hop_length=hop_length, pad_l=pad,
                                                      pad_r=pad, discard_dc=False)
        waves_mel_recon_mel_recon = spectral_ops.stfts_to_waves(stfts_mel_recon_mel_recon, n_fft=n_fft,
                                                                hop_length=hop_length, pad_l=pad, pad_r=pad,
                                                                discard_dc=False)
        assert (tf.abs(tf.reduce_mean(tf.abs(waves_mel_recon) - tf.abs(waves_mel_recon_mel_recon))).numpy() < 0.05)

        # TODO: FIX NUMERICAL ERRORS!
        # This performance is quite terrible, check if possible to fix the IFFT somehow...
        # Update: Cast to tf.float64 before fft/ifft to get rid of most of numerical error!
        #
        assert (tf.__version__ != '2.1.0')

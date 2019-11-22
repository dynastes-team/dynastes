import os

import tensorflow as tf
from tensorflow.python.framework import test_util

from dynastes.ops import spectral_ops


class SpectralOpsTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):

        print('### Running spectral ops tests ###')

        n_fft = 1024
        fft_bins = n_fft // 2
        hop_length = 256
        pad = 8192 * 4
        mel_upsampling = 8
        dir_path = os.path.dirname(os.path.realpath(__file__))

        raw_audio = tf.io.read_file(dir_path + '/data/test_wav.wav')
        waves = tf.audio.decode_wav(raw_audio)[0][:20480] * 0.25
        waves = tf.expand_dims(waves, 0)
        waves = tf.expand_dims(waves, -1)
        waves = tf.keras.layers.UpSampling2D((2, 1), interpolation='bilinear')(waves)
        waves = tf.squeeze(waves, axis=-1)

        stfts = spectral_ops.waves_to_stfts(waves, n_fft=n_fft, hop_length=hop_length, pad_l=pad, pad_r=pad,
                                            discard_dc=True)
        stfts_mags = tf.maximum(tf.minimum(tf.abs(stfts), 0.5), -0.5)
        stfts_angles = tf.math.angle(stfts)
        stfts = spectral_ops.polar2rect(stfts_mags, stfts_angles)

        waves_recon = spectral_ops.stfts_to_waves(stfts, n_fft=n_fft, hop_length=hop_length, pad_l=pad, pad_r=pad,
                                                  discard_dc=True)
        wave_err = tf.abs(tf.reduce_mean(tf.abs(waves) - tf.abs(waves_recon))).numpy()
        wave_dc_err = tf.abs(tf.reduce_mean(waves - waves_recon)).numpy()
        print('wave_err', wave_err, 'wave_dc_err', wave_dc_err)
        assert (wave_err < 0.03)
        assert (wave_dc_err < 0.001)
        waves_recon_stft = spectral_ops.waves_to_stfts(waves_recon, n_fft=n_fft, hop_length=hop_length, pad_l=pad,
                                                       pad_r=pad, discard_dc=True)
        stft_err = tf.abs(tf.reduce_mean(tf.abs(stfts) - tf.abs(waves_recon_stft))).numpy()
        stft_mean_err = tf.abs(tf.reduce_mean(stfts - waves_recon_stft)).numpy()
        print('stft_err', stft_err, 'stft_mean_err', stft_mean_err)
        assert (stft_err < 0.005)
        assert (stft_mean_err < 1e-4)
        l2mel = tf.signal.linear_to_mel_weight_matrix(num_spectrogram_bins=fft_bins, num_mel_bins=(fft_bins * mel_upsampling))
        # Mel transform
        melspecgrams = spectral_ops.stfts_to_melspecgrams(stfts, l2mel=l2mel)

        # Using mel-upsampling yields much lower reconstruction error!
        if mel_upsampling > 1:
            melspecgrams = tf.keras.layers.AveragePooling2D((1, mel_upsampling), strides=(1, mel_upsampling), padding='same', data_format='channels_last')(melspecgrams)
            # Simulate output downsampling
            melspecgrams = tf.keras.layers.UpSampling2D((1, mel_upsampling), data_format='channels_last', interpolation='bilinear')(melspecgrams)

        stfts_mel_recon = spectral_ops.melspecgrams_to_stfts(melspecgrams, mel2l=tf.transpose(l2mel))
        waves_mel_recon = spectral_ops.stfts_to_waves(stfts_mel_recon, n_fft=n_fft, hop_length=hop_length, discard_dc=True, pad_l=pad, pad_r=pad)
        melspec_wav_recon_err = tf.abs(tf.reduce_mean(waves - waves_mel_recon)).numpy()
        print('melspec_wav_recon_err', melspec_wav_recon_err)
        assert(melspec_wav_recon_err < 0.0002)


        melspecgram_stfts_mel_recon = spectral_ops.stfts_to_melspecgrams(stfts_mel_recon, l2mel=l2mel)

        melspecgrams_err = tf.abs(tf.reduce_mean(tf.abs(melspecgrams) - tf.abs(melspecgram_stfts_mel_recon))).numpy()
        melspecgrams_mean_err = tf.abs(tf.reduce_mean(melspecgrams - melspecgram_stfts_mel_recon)).numpy()
        print('melspecgrams_err', melspecgrams_err, 'melspecgrams_mean_err', melspecgrams_mean_err)
        assert (melspecgrams_err < 0.02)
        assert (melspecgrams_mean_err < 0.021)

        stfts_mel_recon_mel_recon = spectral_ops.melspecgrams_to_stfts(melspecgram_stfts_mel_recon,
                                                                       mel2l=tf.transpose(l2mel))
        mel_recon_err = tf.abs(tf.reduce_mean(tf.abs(stfts_mel_recon) - tf.abs(stfts_mel_recon_mel_recon))).numpy()
        print('mel_recon_err', mel_recon_err)
        assert (mel_recon_err < 0.001)
        waves_mel_recon = spectral_ops.stfts_to_waves(stfts_mel_recon, n_fft=n_fft, hop_length=hop_length, pad_l=pad,
                                                      pad_r=pad, discard_dc=True)
        waves_mel_recon_mel_recon = spectral_ops.stfts_to_waves(stfts_mel_recon_mel_recon, n_fft=n_fft,
                                                                hop_length=hop_length, pad_l=pad, pad_r=pad,
                                                                discard_dc=True)
        assert (tf.abs(tf.reduce_mean(tf.abs(waves_mel_recon) - tf.abs(waves_mel_recon_mel_recon))).numpy() < 0.05)

        # TODO: FIX/IMPROVE NUMERICAL ERRORS!
        # This performance is quite terrible, check if possible to fix the IFFT somehow...
        # Update: Cast to tf.float64 before fft/ifft to get rid of most of numerical error!
        #
        assert (tf.__version__ != '2.1.0')

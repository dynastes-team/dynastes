import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util

from dynastes.layers.spectral_layers import Wave2STFTLayer, STFT2MelspectrogramLayer
from dynastes.util.test_utils import layer_test


class SpectralLayersTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        print('### Running spectral ops tests ###')
        for fft_bins in [128, 256, 512]:
            for hop_div in [2, 4, 8]:
                hop_length = fft_bins // hop_div
                for pad_mul in [2, 3, 4, 7]:
                    pad = fft_bins * pad_mul
                    n_mels = 128
                    dir_path = os.path.dirname(os.path.realpath(__file__))

                    raw_audio = tf.io.read_file(dir_path + '/data/test_wav.wav')
                    waves = tf.audio.decode_wav(raw_audio)[0][:1024] * 0.25
                    waves = tf.expand_dims(waves, 0)
                    waves = tf.expand_dims(waves, -1)
                    waves = tf.keras.layers.UpSampling2D((2, 1), interpolation='bilinear')(waves)
                    waves: tf.Tensor = tf.squeeze(waves, axis=-1)

                    wav2stft_layer = Wave2STFTLayer(n_bins=fft_bins,
                                                    hop_length=hop_length,
                                                    padding=pad)

                    stft2mel_layer = STFT2MelspectrogramLayer(n_mels=n_mels,
                                                              min_hz=40.,
                                                              max_hz=16000.,
                                                              sr=44100,
                                                              ifreq=True,
                                                              return_phase=True,
                                                              trainable=True)
                    with tf.GradientTape() as t:

                        stfts = wav2stft_layer(waves)

                        comp_stfts_out_shape = wav2stft_layer.compute_output_shape(waves.shape.as_list())
                        stfts_shape = stfts.shape.as_list()
                        try:
                            self.assertEqual(stfts_shape, comp_stfts_out_shape)
                        except AssertionError as err:
                            print(fft_bins, hop_length, pad)
                            raise err

                        melspecs = stft2mel_layer(stfts)
                        comp_melspecs_out_shape = stft2mel_layer.compute_output_shape(stfts_shape)
                        melspecs_shape = melspecs.shape.as_list()
                        self.assertEqual(comp_melspecs_out_shape, melspecs_shape)

                        istfts = stft2mel_layer(melspecs, invert=True)
                        iwaves = wav2stft_layer(istfts, invert=True)

                        loss = tf.keras.losses.mean_absolute_error(waves, iwaves)

                    grads = t.gradient(loss, stft2mel_layer.trainable_weights)
                    for grad in grads:
                        self.assertNotAllClose(grad, np.zeros_like(grad))


class Wav2STFTLayerTest(tf.test.TestCase):

    def test_simple(self):
        layer_test(
            Wave2STFTLayer, kwargs={'n_bins': 128, 'hop_length': 64, 'padding': 64, 'hq': True},
            input_shape=(5, 32, 2),
            expected_output_dtype=tf.complex128)


class STFT2MelSpecLayerTest(tf.test.TestCase):

    def test_simple(self):
        layer_test(
            STFT2MelspectrogramLayer, kwargs={'n_mels': 128},
            input_shape=(5, 32, 128, 2),
            input_dtype=tf.complex128,
            expected_output_dtype=tf.float32)

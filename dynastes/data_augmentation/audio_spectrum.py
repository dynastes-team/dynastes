import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp

from dynastes.ops.t2t_common import shape_list


def sparse_warp(mel_spectrograms, time_warping_para: float = 80.):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    Args:
      mel_spectrograms: Tensor of log magnitudes and possibly instantaneous frequencies,
            shape [..., time, freq, ch*(1/2)], mel scaling of frequencies.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
    Returns:
      mel_spectrograms: Tensor of log magnitudes and possibly instantaneous frequencies,
            shape [..., time, freq, ch*(1/2)], mel scaling of frequencies.
    """

    fbank_size = shape_list(mel_spectrograms)
    _, n, n_mels, _ = fbank_size
    # Step 1 : Time warping
    # Image warping control point setting.
    # Source
    pt = tf.random.uniform([], 0, n - (time_warping_para * 2),
                           tf.float32) + time_warping_para  # radnom point along the time axis
    src_ctr_pt_freq = tf.cast(tf.range(n_mels // 2), tf.float32)  # control points on freq-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
    src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

    # Destination
    w = tf.random.uniform([], -time_warping_para, time_warping_para, tf.float32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

    # warp
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)
    warped_image, _ = sparse_image_warp(mel_spectrograms,
                                        source_control_point_locations,
                                        dest_control_point_locations)
    return warped_image


def frequency_masking(mel_spectrograms, frequency_masking_para: int = 100, frequency_mask_num: int = 1, roll_mask=None):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    Args:
      mel_spectrograms: Tensor of log magnitudes and possibly instantaneous frequencies,
            shape [..., time, freq, ch*(1/2)], mel scaling of frequencies.
      frequency_masking_para(int): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      frequency_mask_num(int): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
    Returns:
      mel_spectrograms: Tensor of log magnitudes and possibly instantaneous frequencies,
            shape [..., time, freq, ch*(1/2)], mel scaling of frequencies.
    """
    # Step 2 : Frequency masking
    fbank_size = shape_list(mel_spectrograms)
    _, n, n_mels, _ = fbank_size
    frequency_masking_para = min(frequency_masking_para, n_mels // 2)

    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=n_mels - f, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0
        mask = tf.concat((tf.ones(shape=(1, n, n_mels - f0 - f, 1)),
                          tf.zeros(shape=(1, n, f, 1)),
                          tf.ones(shape=(1, n, f0, 1)),
                          ), 2)
        if roll_mask is not None:
            roll_mel_spectrograms = tf.roll(mel_spectrograms, roll_mask, axis=0)
            mel_spectrograms = (mel_spectrograms * mask) + (roll_mel_spectrograms * (1-mask))
        else:
            mel_spectrograms = mel_spectrograms * mask
    return tf.cast(mel_spectrograms, dtype=tf.float32)


def time_masking(mel_spectrograms, time_masking_para: int = 27, time_mask_num: int = 1, roll_mask=None):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    Args:
      mel_spectrograms(tf.Tensor): Tensor of log magnitudes and possibly instantaneous frequencies / phases,
            shape [..., time, freq, ch*(1/2)], mel scaling of frequencies.
      time_masking_para(int): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      time_mask_num(int): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    Returns:
      mel_spectrogram: Tensor of log magnitudes and possibly instantaneous frequencies,
            shape [..., time, freq, ch*(1/2)], mel scaling of frequencies.
    """
    fbank_size = shape_list(mel_spectrograms)
    _, n, n_mels, _ = fbank_size
    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=n - t, dtype=tf.int32)

        # mel_spectrograms[:, t0:t0 + t] = 0
        mask = tf.concat((tf.ones(shape=(1, n - t0 - t, n_mels, 1)),
                          tf.zeros(shape=(1, t, n_mels, 1)),
                          tf.ones(shape=(1, t0, n_mels, 1)),
                          ), 1)
        if roll_mask is not None:
            roll_mel_spectrograms = tf.roll(mel_spectrograms, roll_mask, axis=0)
            mel_spectrograms = (mel_spectrograms * mask) + (roll_mel_spectrograms * (1-mask))
        else:
            mel_spectrograms = mel_spectrograms * mask

    return tf.cast(mel_spectrograms, dtype=tf.float32)


def spec_augment(mel_spectrograms: tf.Tensor,
                 time_warping_para: float = 80.,
                 time_masking_para: int = 27,
                 time_mask_num: int = 1,
                 frequency_masking_para: int = 100,
                 frequency_mask_num: int = 1,
                 normalize=True,
                 roll_mask=None):
    """
    Args:
      mel_spectrograms(tf.Tensor): Tensor of log magnitudes and possibly instantaneous frequencies / phases,
            shape [..., time, freq, ch*(1/2)], mel scaling of frequencies.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      time_masking_para(int): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      time_mask_num(int): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
      frequency_masking_para(int): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      frequency_mask_num(int): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      roll_mask(int): if not none, replace masked indices with batch rolled by [roll_mask] steps
    Returns:
      mel_spectrograms: Tensor of log magnitudes and possibly instantaneous frequencies,
            shape [..., time, freq, ch*(1/2)], mel scaling of frequencies.
    """
    if normalize:
        mean, std = tf.nn.moments(mel_spectrograms, axes=[0, 1], keepdims=True)
        mel_spectrograms = (mel_spectrograms - mean) / std

    warped_mel_spectrogram = sparse_warp(mel_spectrograms,
                                         time_warping_para=time_warping_para)

    warped_frequency_spectrogram = frequency_masking(warped_mel_spectrogram,
                                                     frequency_masking_para=frequency_masking_para,
                                                     frequency_mask_num=frequency_mask_num,
                                                     roll_mask=roll_mask)

    warped_frequency_time_spectrogram = time_masking(warped_frequency_spectrogram,
                                                     time_masking_para=time_masking_para,
                                                     time_mask_num=time_mask_num,
                                                     roll_mask=roll_mask)

    if normalize:
        warped_frequency_time_spectrogram = (warped_frequency_time_spectrogram * std) + mean
    return warped_frequency_time_spectrogram

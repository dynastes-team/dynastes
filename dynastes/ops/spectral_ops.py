import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import window_ops

from dynastes.core.nn import array_ops as d_array_ops

"""
    STFT, Mel-spectrogram and inverse methods
    Borrows heavily from
    https://github.com/tensorflow/magenta/tree/master/magenta/models/gansynth/lib
    But supports multi-channel input, ie: [..., time, channels] -> [..., time, mels, channels * 2]
"""

def inverse_stft_window_fn(frame_step,
                           forward_window_fn=window_ops.hann_window,
                           name=None):
    """Generates a window function that can be used in `inverse_stft`.
    Constructs a window that is equal to the forward window with a further
    pointwise amplitude correction.  `inverse_stft_window_fn` is equivalent to
    `forward_window_fn` in the case where it would produce an exact inverse.
    See examples in `inverse_stft` documentation for usage.
    Args:
      frame_step: An integer scalar `Tensor`. The number of samples to step.
      forward_window_fn: window_fn used in the forward transform, `stft`.
      name: An optional name for the operation.
    Returns:
      A callable that takes a window length and a `dtype` keyword argument and
        returns a `[window_length]` `Tensor` of samples in the provided datatype.
        The returned window is suitable for reconstructing original waveform in
        inverse_stft.
    """
    with ops.name_scope(name, 'inverse_stft_window_fn', [forward_window_fn]):
        frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
        frame_step.shape.assert_has_rank(0)

    def inverse_stft_window_fn_inner(frame_length, dtype):
        """Computes a window that can be used in `inverse_stft`.
        Args:
          frame_length: An integer scalar `Tensor`. The window length in samples.
          dtype: Data type of waveform passed to `stft`.
        Returns:
          A window suitable for reconstructing original waveform in `inverse_stft`.
        Raises:
          ValueError: If `frame_length` is not scalar, `forward_window_fn` is not a
          callable that takes a window length and a `dtype` keyword argument and
          returns a `[window_length]` `Tensor` of samples in the provided datatype
          `frame_step` is not scalar, or `frame_step` is not scalar.
        """
        with ops.name_scope(name, 'inverse_stft_window_fn', [forward_window_fn]):
            frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
            frame_length.shape.assert_has_rank(0)

            # Use equation 7 from Griffin + Lim.
            forward_window = forward_window_fn(frame_length, dtype=dtype)
            denom = math_ops.square(forward_window)
            overlaps = -(-frame_length // frame_step)  # Ceiling division.
            denom = array_ops.pad(denom, [(0, overlaps * frame_step - frame_length)])
            denom = array_ops.reshape(denom, [overlaps, frame_step])
            denom = math_ops.reduce_sum(denom, 0, keepdims=True)
            denom = array_ops.tile(denom, [overlaps, 1])
            denom = array_ops.reshape(denom, [overlaps * frame_step])
            denom = tf.maximum(denom, 1e-16)
            return forward_window / denom[:frame_length]

    return inverse_stft_window_fn_inner


def eps(tensor_dtype: tf.DType):
    return tf.convert_to_tensor(np.finfo(tensor_dtype.as_numpy_dtype).eps, dtype=tensor_dtype) * 10000


def _safe_log(x: tf.Tensor):
    return tf.math.log(x + eps(x.dtype))


def waves_to_stfts(waves: tf.Tensor, n_fft=512, hop_length=256, discard_dc=True, pad_l=128, pad_r=128) -> tf.Tensor:
    """Convert from waves to complex stfts.
       Args:
         waves: Tensor of the waveform, shape [..., time, channels].
       Returns:
         stfts: Complex64 tensor of stft, shape [..., time, freq, channels].
       """
    stfts = _waves_to_stfts(waves, n_fft=n_fft, hop_length=hop_length, discard_dc=discard_dc, pad_l=pad_l, pad_r=pad_r)
    stfts = tf.squeeze(stfts, axis=-1)  # [..., channels, time, freq]
    stfts_shape = stfts.shape.as_list()
    perm = list(range(len(stfts_shape)))
    perm = perm[:-3] + [perm[-2], perm[-1], perm[-3]]
    return tf.transpose(stfts, perm=perm)


def stfts_to_waves(stfts: tf.Tensor, n_fft=512, hop_length=256, discard_dc=True, pad_l=128, pad_r=128) -> tf.Tensor:
    """Convert from complex stfts to waves.
    Args:
      stfts: Complex64 tensor of stft, shape [..., time, freq, channels].
    Returns:
      waves: Tensor of the waveform, shape [..., time, channels].
    """
    stfts_shape = stfts.shape.as_list()
    perm = list(range(len(stfts_shape)))
    perm = perm[:-3] + [perm[-1], perm[-3], perm[-2]]
    stfts = tf.transpose(stfts, perm=perm)  # [..., channels, time, freq]
    stfts = tf.expand_dims(stfts, axis=-1)

    waves = _stfts_to_waves(stfts, n_fft=n_fft, hop_length=hop_length, discard_dc=discard_dc, pad_l=pad_l,
                            pad_r=pad_r)  # [..., channels, time, freq]
    return waves


def stfts_to_melspecgrams(stfts: tf.Tensor, l2mel, ifreq=True) -> tf.Tensor:
    """Converts stfts to specgrams.
    Args:
      stfts: Complex64 tensor of stft, shape [..., time, freq, channels].
    Returns:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [..., time, freq, 2*channels], mel scaling of frequencies.
    """
    # inp: [..., time, freq, channels]
    stfts_shape = stfts.shape.as_list()
    perm = list(range(len(stfts_shape)))
    perm = perm[:-3] + [perm[-1], perm[-3], perm[-2]]
    stfts = tf.transpose(stfts, perm=perm)
    stfts = tf.expand_dims(stfts, axis=-1)  # [..., channels, time, freq, 1]
    melspecgrams = _stfts_to_melspecgrams(stfts, l2mel=l2mel, ifreq=ifreq)  # [..., channels, time, freq, 2]

    melspecgrams_shape = melspecgrams.shape.as_list()
    perm = list(range(len(melspecgrams_shape)))
    perm = perm[:-4] + [perm[-3], perm[-2], perm[-4], perm[-1]]
    melspecgrams = tf.transpose(melspecgrams, perm=perm)  # [..., time, freq, channels, 2]
    melspecgrams_shape = melspecgrams.shape.as_list()
    melspecgrams = tf.reshape(melspecgrams, melspecgrams_shape[:-2] + [
        melspecgrams_shape[-2] * melspecgrams_shape[-1]])  # [..., time, freq, channels * 2]
    return melspecgrams


def melspecgrams_to_stfts(melspecgrams: tf.Tensor, mel2l, ifreq=True) -> tf.Tensor:
    """Converts melspecgrams to stfts.
        Args:
          melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
            shape [..., time, freq, 2*channels], mel scaling of frequencies.
          mel2l: Mel to linear matrix, ie transposed linear to mel matrix
            @see
            tf.signal.linear_to_mel_weight_matrix
        Returns:
          specgrams: Tensor of log magnitudes and instantaneous frequencies,
            shape [..., time, freq, channels].
        """
    melspecgrams_shape = melspecgrams.shape.as_list()  # [..., time, freq, channels*2]
    melspecgrams = tf.reshape(melspecgrams,
                              melspecgrams_shape[:-1] + [melspecgrams_shape[-1] // 2,
                                                         2])  # [..., time, freq, channels, 2]
    perm = list(range(len(melspecgrams_shape) + 1))
    perm = perm[:-4] + [perm[-2], perm[-4], perm[-3], perm[-1]]
    melspecgrams = tf.transpose(melspecgrams, perm=perm)  # [..., channels, time, freq, 2]
    stfts = _melspecgrams_to_stfts(melspecgrams, mel2l=mel2l, ifreq=True)  # [..., channels, time, freq, 1]
    stfts = tf.squeeze(stfts, axis=-1)  # [..., channels, time, freq]
    stfts_shape = stfts.shape.as_list()
    perm = list(range(len(stfts_shape)))
    perm = perm[:-3] + [perm[-2], perm[-1], perm[-3]]
    stfts = tf.transpose(stfts, perm=perm)  # [..., time, freq, channels]
    return stfts


def _waves_to_stfts(waves: tf.Tensor, n_fft=512, hop_length=256, discard_dc=True, pad_l=128, pad_r=128) -> tf.Tensor:
    """Convert from waves to complex stfts.
    Args:
      waves: Tensor of the waveform, shape [..., time, channels].
    Returns:
      stfts: Complex64 tensor of stft, shape [..., channels, time, freq, 1].
    """
    waves_shape = waves.shape.as_list()
    waves = tf.linalg.matrix_transpose(waves)  # [..., channels, time]
    waves_padded = tf.pad(waves,
                          np.reshape(np.asarray([0, 0] * (len(waves_shape) - 1)), (-1, 2)).tolist() + [[pad_l, pad_r]])
    stfts = tf.signal.stft(
        waves_padded,
        window_fn=tf.signal.hann_window,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        pad_end=False)
    if discard_dc:
        stfts, dc = tf.split(stfts, num_or_size_splits=[n_fft // 2, 1], axis=-1)
    return tf.expand_dims(stfts, axis=-1)


def _stfts_to_waves(stfts, n_fft=512, hop_length=256, discard_dc=True, pad_l=128, pad_r=128):
    """Convert from complex stfts to waves.
    Args:
      stfts: Complex64 tensor of stft, shape [..., channels, time, freq, 1].
    Returns:
      waves: Tensor of the waveform, shape [..., time, channels].
    """
    stfts = tf.squeeze(stfts, axis=-1)
    stfts_shape = stfts.shape.as_list()
    dc = 1 if discard_dc else 0
    nyq = 1 - dc
    stfts = tf.pad(stfts, np.reshape(np.asarray([0, 0] * (len(stfts_shape) - 1)), (-1, 2)).tolist() + [[dc, nyq]])
    waves_resyn = tf.signal.inverse_stft(
        stfts=stfts,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=inverse_stft_window_fn(frame_step=hop_length))
    waves_resyn = tf.linalg.matrix_transpose(waves_resyn)
    crops = np.reshape(np.asarray([0, 0] * (len(stfts_shape) - 3)), (-1, 2)).tolist() + [[pad_l, pad_r], [0, 0]]
    return d_array_ops.crop(waves_resyn, crops)


def _stfts_to_specgrams(stfts, ifreq=True):
    """Converts stfts to specgrams.
    Args:
      stfts: Complex64 tensor of stft, shape [..., channels, time, freq, 1].
    Returns:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [..., channels, time, freq, 2].
    """
    stfts = tf.squeeze(stfts, axis=-1)
    logmag = _safe_log(tf.abs(stfts))

    phase_angle = tf.math.angle(stfts)
    if ifreq:
        p = instantaneous_frequency(phase_angle)
    else:
        p = phase_angle / np.pi

    return tf.stack([logmag, p], axis=-1)


def _specgrams_to_melspecgrams(specgrams, l2mel):
    """Converts specgrams to melspecgrams.
    Args:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [..., channels, time, freq, 2].
      l2mel: Linear to mel matrix
    Returns:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [..., channels, time, freq, 2], mel scaling of frequencies.
    """

    logmag, p = tf.unstack(specgrams, axis=-1)

    mag2 = tf.exp(2.0 * logmag)
    phase_angle = tf.cumsum(p * np.pi, axis=-2)

    logmelmag2 = _safe_log(tf.tensordot(mag2, l2mel, 1))
    mel_phase_angle = tf.tensordot(phase_angle, l2mel, 1)
    mel_p = instantaneous_frequency(mel_phase_angle)

    return tf.stack([logmelmag2, mel_p], axis=-1)


def _stfts_to_melspecgrams(stfts: tf.Tensor, l2mel, ifreq=True) -> tf.Tensor:
    """Converts stfts to specgrams.
    Args:
      stfts: Complex64 tensor of stft, shape [..., channels, time, freq, 1].
    Returns:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [..., channels, time, freq, 2], mel scaling of frequencies.
    """
    logmelmag2 = _safe_log(tf.matmul(tf.abs(stfts), l2mel, transpose_a=True))
    logmelmag2 = tf.linalg.matrix_transpose(logmelmag2)
    phase_angle = tf.math.angle(stfts)
    mel_phase_angle = tf.matmul(phase_angle, l2mel, transpose_a=True)
    mel_phase_angle = tf.linalg.matrix_transpose(mel_phase_angle)
    if ifreq:
        mel_p = instantaneous_frequency(mel_phase_angle)
    else:
        mel_p = mel_phase_angle / np.pi
    return tf.concat([logmelmag2, mel_p], axis=-1)


def _specgrams_to_stfts(specgrams: tf.Tensor, ifreq=True) -> tf.Tensor:
    """Converts specgrams to stfts.
    Args:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [..., channels, time, freq, 2].
    Returns:
      stfts: Complex64 tensor of stft, shape [..., channels, time, freq, 1].
    """
    logmag, p = tf.unstack(specgrams, axis=-1)

    mag = tf.exp(logmag)

    if ifreq:
        phase_angle = tf.cumsum(p * np.pi, axis=-2)
    else:
        phase_angle = p * np.pi

    return tf.expand_dims(polar2rect(mag, phase_angle), axis=-1)


def _melspecgrams_to_specgrams(melspecgrams: tf.Tensor, mel2l) -> tf.Tensor:
    """Converts melspecgrams to specgrams.
    Args:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [..., channels, time, freq, 2], mel scaling of frequencies.
      mel2l: Mel to linear matrix
    Returns:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [..., channels, time, freq, 2].
    """

    logmelmag2, mel_p = tf.unstack(melspecgrams, axis=-1)

    mag2 = tf.tensordot(tf.exp(logmelmag2), mel2l, 1)
    logmag = 0.5 * _safe_log(mag2)
    mel_phase_angle = tf.cumsum(mel_p * np.pi, axis=-2)
    phase_angle = tf.tensordot(mel_phase_angle, mel2l, 1)
    p = instantaneous_frequency(phase_angle)

    return tf.stack([logmag, p], axis=-1)


def _melspecgrams_to_stfts(melspecgrams, mel2l, ifreq=True):
    """Converts melspecgrams to specgrams.
        Args:
          melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
            shape [..., channels, time, freq, 2], mel scaling of frequencies.
          mel2l: Mel to linear matrix
        Returns:
          stfts: Tensor of log magnitudes and instantaneous frequencies,
            shape [..., channels, time, freq, 1].
        """
    logmelmag, mel_p = tf.unstack(melspecgrams, axis=-1)

    mag = tf.matmul(tf.exp(logmelmag), mel2l)
    if ifreq:
        mel_phase_angle = tf.cumsum(mel_p * np.pi, axis=-2)
    else:
        mel_phase_angle = mel_p * np.pi

    phase_angle = tf.matmul(mel_phase_angle, mel2l)

    return tf.expand_dims(polar2rect(mag, phase_angle), axis=-1)


def _diff(x, axis=-1):
    """Take the finite difference of a tensor along an axis.
    Args:
      x: Input tensor of any dimension.
      axis: Axis on which to take the finite difference.
    Returns:
      d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
      ValueError: Axis out of range for tensor.
    """
    shape = x.get_shape()
    if axis >= len(shape):
        raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                         (axis, len(shape)))

    begin_back = [0 for _ in range(len(shape))]
    begin_front = [0 for _ in range(len(shape))]
    begin_front[axis] = 1

    size = shape.as_list()
    size[axis] -= 1
    slice_front = tf.slice(x, begin_front, size)
    slice_back = tf.slice(x, begin_back, size)
    d = slice_front - slice_back
    return d


def _unwrap(p, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
      p: Phase tensor.
      discont: Float, size of the cyclic discontinuity.
      axis: Axis of which to unwrap.
    Returns:
      unwrapped: Unwrapped tensor of same size as input.
    """
    dd = _diff(p, axis=axis)
    ddmod = tf.math.mod(dd + np.pi, 2.0 * np.pi) - np.pi
    idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
    ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
    ph_correct = ddmod - dd
    ph_cumsum = tf.cumsum(ph_correct, axis=axis)

    shape = p.get_shape().as_list()
    shape[axis] = 1
    ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
    return unwrapped


def instantaneous_frequency(phase_angle, time_axis=-2, use_unwrap=True):
    """Transform a fft tensor from phase angle to instantaneous frequency.
    Take the finite difference of the phase. Pad with initial phase to keep the
    tensor the same size.
    Args:
      phase_angle: Tensor of angles in radians. [..., Time, Freqs]
      time_axis: Axis over which to unwrap and take finite difference.
      use_unwrap: True preserves original GANSynth behavior, whereas False will
          guard against loss of precision.
    Returns:
      dphase: Instantaneous frequency (derivative of phase). Same size as input.
    """
    if use_unwrap:
        # Can lead to loss of precision.
        phase_unwrapped = _unwrap(phase_angle, axis=time_axis)
        dphase = _diff(phase_unwrapped, axis=time_axis)
    else:
        # Keep dphase bounded. N.B. runs faster than a single mod-2pi expression.
        dphase = _diff(phase_angle, axis=time_axis)
        dphase = tf.where(dphase > np.pi, dphase - 2 * np.pi, dphase)
        dphase = tf.where(dphase < -np.pi, dphase + 2 * np.pi, dphase)

    # Add an initial phase to dphase.
    size = phase_angle.get_shape().as_list()
    size[time_axis] = 1
    begin = [0 for _ in size]
    phase_slice = tf.slice(phase_angle, begin, size)
    dphase = tf.concat([phase_slice, dphase], axis=time_axis) / np.pi
    return dphase


def polar2rect(mag, phase_angle):
    """Convert polar-form complex number to its rectangular form."""
    mag = tf.complex(mag, tf.convert_to_tensor(0.0, dtype=mag.dtype))
    phase = tf.complex(tf.cos(phase_angle), tf.sin(phase_angle))
    return mag * phase

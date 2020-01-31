import tensorflow as tf

from dynastes.data_augmentation.audio_spectrum import spec_augment
from dynastes.layers.base_layers import DynastesBaseLayer


@tf.keras.utils.register_keras_serializable(package='Dynastes')
class SpecAugmentLayer(DynastesBaseLayer):

    def __init__(self,
                 time_warping_para: float = 80.,
                 time_masking_para: int = 27,
                 time_mask_num: int = 1,
                 frequency_masking_para: int = 100,
                 frequency_mask_num: int = 1,
                 normalize=True,
                 roll_mask=None,
                 **kwargs):
        kwargs['dynamic'] = True
        super(SpecAugmentLayer, self).__init__(**kwargs)
        self.time_warping_para: float = time_warping_para
        self.time_masking_para: int = time_masking_para
        self.time_mask_num: int = time_mask_num
        self.frequency_masking_para: int = frequency_masking_para
        self.frequency_mask_num: int = frequency_mask_num
        self.normalize = normalize
        self.roll_mask = roll_mask

    def call(self, inputs, training=None):
        if training:
            return spec_augment(inputs,
                                time_warping_para=self.time_warping_para,
                                time_masking_para=self.time_masking_para,
                                time_mask_num=self.time_mask_num,
                                frequency_masking_para=self.frequency_masking_para,
                                frequency_mask_num=self.frequency_mask_num,
                                normalize=self.normalize,
                                roll_mask=self.roll_mask)
        else:
            return inputs

    def get_config(self):
        config = {
            'time_warping_para': self.time_warping_para,
            'time_masking_para': self.time_masking_para,
            'time_mask_num': self.time_mask_num,
            'frequency_masking_para': self.frequency_masking_para,
            'frequency_mask_num': self.frequency_mask_num,
            'normalize': self.normalize,
            'roll_mask': self.roll_mask,
        }
        base_config = super(SpecAugmentLayer, self).get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

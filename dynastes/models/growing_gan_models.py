import abc

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def _call_masked(layer, inputs, training=None, mask=None, **kwargs):
    if layer.supports_masking:
        out = layer(inputs, training=training, mask=mask, **kwargs)
        out_mask = layer.compute_mask(inputs, mask, **kwargs)
    else:
        out = layer(inputs, training=training, **kwargs)
        out_mask = mask
    return out, out_mask


class GrowingGanModel(tfk.Model, abc.ABC):
    @abc.abstractmethod
    def interpolate_domain(self, x, y, interp):
        """
        Return result of interpolating between domains
        """

    @staticmethod
    def interpolate_mask(x_mask, y_mask, interp):

        @tf.function
        def interp_masks(_x_mask, _y_mask, _interp):

            if _interp <= 0.:
                return _x_mask
            elif _interp >= 1.:
                return _y_mask
            else:
                return tf.math.logical_and(_x_mask, _y_mask)

        if x_mask is None:
            return y_mask
        else:
            if y_mask is None:
                return x_mask
            else:
                return interp_masks(x_mask, y_mask, interp)


class GrowingGanGenerator(GrowingGanModel, abc.ABC):
    def __init__(self,
                 n_lods=0,
                 **kwargs):
        super(GrowingGanGenerator, self).__init__(**kwargs)
        self.n_lods = n_lods

    """
    @tf.function
    def get_images_ag(x, lod_in):

        def grow(x, res, lod_in):
            print("grow, res:", 2 ** res, ' lods left: ', lod_in)
            y = block(res, x)

            def get_lod_out():
                y_rgb = torgb(res, y)  # This stage's output in rgb
                if lod_in > lod_in:  # Currently lerping
                    x_rgb = upscale2d(torgb(res - 1, x))
                    z = lerp(y_rgb, x_rgb, lod_in - lod_in)
                else:  # No need to lerp at this stage
                    z = y_rgb
                return upscale2d(z, 2 ** lod_in)  ##Conform to output

            if lod_in > 0:
                if lod_in < lod_in:  # Lod decreases per rec until we're at lod_in
                    return grow(y, res + 1, lod_in - 1)
                else:  # If this stage should output
                    return get_lod_out()
            else:  # We're at lod_in 0 so we return the "lod_in-stack", this would be the innermost call of the recursion loop
                return get_lod_out()

        # Stylegan starts at lod_in 3 = 16x16
        print('resolution_log2', resolution_log2)
        images_out = grow(x, 3, resolution_log2 - 3)
        return images_out
    """

    @abc.abstractmethod
    def get_gan_layer_by_lod(self, lod) -> tfkl.Layer:
        """ Return processing block here """

    @abc.abstractmethod
    def get_to_domain_layer_by_lod(self, lod) -> tfkl.Layer:
        """
        Return a layer that transforms output of
        gan_layer at this lod_in into a tensor that
        is compatible with your output domain
        The domain should be the same for every
        lod_in, but doesn't need to be RGB etc, if
        such a conversion is handled by the layer returned
        by get_conform_to_output_layer()
        Lerping between lods is handled in this "domain"
        """

    @abc.abstractmethod
    def get_upscale_domain_layer_by_lod(self, input_lod, output_lod) -> tfkl.Layer:
        """
        Return a layer that scales input from input_lod
        to output lod_in dimensions, this happens in
        the "domain" space

        caveats:
        Possibly you might have to perform cumprod
        on hparams strides, depending on your architecture
        if it's a simple one, you might just return Upsampling2D here

        """

    @abc.abstractmethod
    def get_conform_to_output_layer_by_lod(self, input_lod) -> tfkl.Layer:
        """
        Return a layer that scales/transforms input @ input_lod
        to conform exactly to targets
        """


    def _get_output(self, inputs, lod_in=None, training=None, **kwargs):

        def grow(gen: GrowingGanGenerator, x, current_lod):

            y_layer = gen.get_gan_layer_by_lod(current_lod)
            y = y_layer(x, training=training, **kwargs)

            lods_left = gen.n_lods - (current_lod + 1)

            def get_lod_output(x, y, current_lod):
                y_domain = gen.get_to_domain_layer_by_lod(current_lod)(y, training=training, **kwargs)
                if lod_in > lods_left:

                    x_domain = gen.get_to_domain_layer_by_lod(current_lod - 1)(x, training=training, **kwargs)

                    x_as_y_domain = gen.get_upscale_domain_layer_by_lod(current_lod - 1, current_lod)(x_domain,
                                                                                                      training=training,
                                                                                                      **kwargs)
                    z = gen.interpolate_domain(y_domain, x_as_y_domain, lod_in - lods_left)
                else:
                    z = y_domain

                return gen.get_conform_to_output_layer_by_lod(current_lod)(z, training=training, **kwargs)

            def ret_fn(x, y, lod_in, lods_left, current_lod):
                if lods_left > 0:
                    if lod_in < lods_left:
                        return grow(gen, y, current_lod=current_lod + 1)
                    else:
                        return get_lod_output(x, y, current_lod)
                else:
                    return get_lod_output(x, y, current_lod)

            return ret_fn(x, y, lod_in, lods_left, current_lod)

        return grow(self, x=inputs, current_lod=0)

    def _get_output_masked(self, inputs, lod_in=None, training=None, mask=None, **kwargs):

        def grow(gen: GrowingGanGenerator, x, current_lod, mask=None):

            y_layer = gen.get_gan_layer_by_lod(current_lod)
            y, y_mask = _call_masked(y_layer, x, training=training, mask=mask, **kwargs)

            lods_left = gen.n_lods - (current_lod + 1)

            def get_lod_output(x, y, y_mask=None, x_mask=None):
                y_domain_layer = gen.get_to_domain_layer_by_lod(current_lod)
                y_domain, y_domain_mask = _call_masked(y_domain_layer, y, training=training, mask=y_mask, **kwargs)

                if lod_in > lods_left:

                    x_domain_layer = gen.get_to_domain_layer_by_lod(current_lod - 1)
                    x_domain, x_domain_mask = _call_masked(x_domain_layer, x, training=training, mask=x_mask, **kwargs)

                    x_to_y_layer = gen.get_upscale_domain_layer_by_lod(current_lod - 1, current_lod)
                    x_as_y_domain, x_as_y_mask = _call_masked(x_to_y_layer, x_domain, training=training,
                                                              mask=x_domain_mask, **kwargs)

                    z = gen.interpolate_domain(y_domain, x_as_y_domain, lod_in - lods_left)
                    z_mask = gen.interpolate_mask(y_domain_mask, x_as_y_mask, lod_in - lods_left)
                else:
                    z = y_domain
                    z_mask = y_domain_mask

                r, r_mask = _call_masked(gen.get_conform_to_output_layer_by_lod(current_lod), z, training=training,
                                         mask=z_mask, **kwargs)
                return r

            if lods_left > 0:
                if lod_in < lods_left:
                    return grow(gen, y, current_lod=current_lod + 1, mask=y_mask)
                else:
                    return get_lod_output(x, y, y_mask=y_mask, x_mask=mask)
            else:
                return get_lod_output(x, y, y_mask=y_mask, x_mask=mask)

        return grow(self, x=inputs, current_lod=0, mask=mask)

    def call(self, inputs, lod_in=None, training=None, mask=None, **kwargs):
        """
        @param lod_in: value between 0. and self.n_lods-1
        @param kwargs: optional arguments passed to every layer on call
        """

        lod_in = tf.maximum(0., tf.minimum(self.n_lods - 1, (self.n_lods - 1) - tf.convert_to_tensor(lod_in)))

        if mask is None:
            return self._get_output(inputs=inputs,
                                    lod_in=lod_in,
                                    training=training,
                                    **kwargs)
        else:
            return self._get_output_masked(inputs=inputs,
                                           lod_in=lod_in,
                                           training=training,
                                           mask=mask, **kwargs)


class GrowingGanClassifier(GrowingGanModel, abc.ABC):
    def __init__(self,
                 n_lods=0,
                 **kwargs):
        super(GrowingGanClassifier, self).__init__(**kwargs)
        self.n_lods = n_lods

    @abc.abstractmethod
    def get_gan_layer_by_lod(self, lod) -> tfkl.Layer:
        """ Return processing block here """

    @abc.abstractmethod
    def get_from_domain_layer_by_lod(self, lod) -> tfkl.Layer:
        """
        Return a layer that transforms input from
        domain into one that can consumed by gan_layer at
        this lod_in.
        The domain should be the same for every
        lod_in, but doesn't need to be RGB etc, if
        such a conversion is handled by the layer returned
        by get_conform_to_output_layer()
        Lerping between lods is handled in this "domain"
        """

    @abc.abstractmethod
    def get_downscale_domain_layer_by_lod(self, input_lod, output_lod) -> tfkl.Layer:
        """
        Return a layer that scales input from input_lod
        to output lod_in dimensions, this happens in
        the "domain" space

        caveats:
        Possibly you might have to perform cumprod
        on hparams strides, depending on your architecture
        if it's a simple one, you might just return AveragePooling2D here

        """

    @abc.abstractmethod
    def get_input_transform_layer_by_lod(self, lod) -> tfkl.Layer:
        """
        Return a layer that scales/transforms input
        to domain @ lod
        """

    def _get_predictions(self, inputs, lod_in, training=None, **kwargs):

        def grow(cls: GrowingGanClassifier, current_lod, lods_left):

            if lods_left > 0 and lod_in < lods_left:
                x = grow(cls, current_lod + 1, lods_left - 1)

                x = cls.get_gan_layer_by_lod(current_lod)(x,
                                                          training=training,
                                                          **kwargs)
            else:
                x = cls.get_input_transform_layer_by_lod(current_lod)(inputs,
                                                                      training=training,
                                                                      **kwargs)
                x = cls.get_from_domain_layer_by_lod(current_lod)(x,
                                                                  training=training,
                                                                  **kwargs)
                x = cls.get_gan_layer_by_lod(current_lod)(x,
                                                          training=training,
                                                          **kwargs)
            if lod_in > lods_left:
                next_x = cls.get_input_transform_layer_by_lod(current_lod - 1)(inputs,
                                                                               training=training,
                                                                               **kwargs)
                next_x = cls.get_from_domain_layer_by_lod(current_lod - 1)(next_x,
                                                                           training=training,
                                                                           **kwargs)
                return cls.interpolate_domain(x, next_x, lod_in - lods_left)
            else:
                return x

        return grow(self, 1, self.n_lods - 1)

    def call(self, inputs, lod_in=None, training=None, mask=None, **kwargs):
        """
        @param lod_in: value between 0. and self.n_lods-1
        @param kwargs: optional arguments passed to every layer on call
        """

        lod_in = tf.maximum(0., tf.minimum(self.n_lods - 1, (self.n_lods - 1) - tf.convert_to_tensor(lod_in)))

        if mask is None:
            return self._get_predictions(inputs=inputs,
                                         lod_in=lod_in,
                                         training=training,
                                         **kwargs)


class SimpleGrowingGanClassifer(GrowingGanClassifier, abc.ABC):

    @abc.abstractmethod
    def get_gan_layer(self, layer_idx) -> tfkl.Layer:
        """
        Return your GAN processing layer
        Latent space needs to be strictly increasing or equal
        @param layer_idx: layer id 0...( n_lods - 1 )
        @type layer_idx: int
        """

    def get_gan_layer_by_lod(self, lod) -> tfkl.Layer:
        if lod < 0:
            raise ArithmeticError
        return self.get_gan_layer((self.n_lods - lod))

    @abc.abstractmethod
    def get_from_domain_layer(self, layer_idx) -> tfkl.Layer:
        """
        Return a layer that converts from your domain-space
        to the latent space at 'layer_idx' index
        Latent space needs to be strictly increasing or equal
        NOTE! This requires one more than your GAN-layers
        @param layer_idx: layer id 0...(n_lods)
        @type layer_idx: int
        """

    def get_from_domain_layer_by_lod(self, lod) -> tfkl.Layer:
        return self.get_from_domain_layer((self.n_lods - lod))

    @abc.abstractmethod
    def get_downscale_domain_by_range(self, input_idx, output_idx) -> tfkl.Layer:

        """
        Return a "in-domain" downscaling layer between your scales, such as:
        self.strides = [2,3,3]
        strides = self.strides[input_idx : output_idx]
        if len(strides) > 0:
            scale = np.cumprod(strides)[-1]
            if scale > 1:
                return tfkl.AveragePooling2D((scale, scale), strides=(scale, scale), padding='same')
        return tfkl.Activation('linear')

        @param input_idx:
        @type input_idx: int
        @param output_idx:
        @type output_idx: int
        """

    def get_downscale_domain_layer_by_lod(self, input_lod, output_lod) -> tfkl.Layer:
        if output_lod == input_lod:
            return tfkl.Activation('linear')
        if input_lod - output_lod == 1:
            return tfkl.Activation('linear')
        return self.get_downscale_domain_by_range(self.n_lods - input_lod, self.n_lods - output_lod)

    def get_input_transform_layer_by_lod(self, lod) -> tfkl.Layer:
        return self.get_downscale_domain_layer_by_lod(self.n_lods, lod)


class SimpleGrowingGanGenerator(GrowingGanGenerator, abc.ABC):

    @abc.abstractmethod
    def get_gan_layer(self, layer_idx) -> tfkl.Layer:
        """
        Return your GAN processing layer
        Latent space needs to be strictly decreasing or equal
        @param layer_idx: layer id 0...( n_lods - 1 )
        @type layer_idx: int
        """

    def get_gan_layer_by_lod(self, lod) -> tfkl.Layer:
        if lod > (self.n_lods - 1):
            return tfkl.Activation('linear')
        return self.get_gan_layer((self.n_lods - 1) - lod)

    @abc.abstractmethod
    def get_to_domain_layer(self, layer_idx) -> tfkl.Layer:
        """
        Return a layer that converts from your latent space
        to your domain space (RGB, what-have-you)
        NOTE! This requires one more than your GAN-layers
        @param layer_idx: layer id 0...(n_lods)
        @type layer_idx: int
        """

    def get_to_domain_layer_by_lod(self, lod) -> tfkl.Layer:
        return self.get_to_domain_layer((self.n_lods - 1) - lod)

    @abc.abstractmethod
    def get_upscale_domain_by_range(self, input_idx, output_idx) -> tfkl.Layer:

        """
        Return a "in-domain" upscale layer between your scales

        Example:

        self.strides = [2,3,3]
        strides = self.strides[input_idx : output_idx]
        if len(strides) > 0:
            scale = np.cumprod(strides)[-1]
            if scale > 1:
                return tfkl.UpSampling2D(size=(scale, scale))
        return tfkl.Activation('linear')

        @param input_idx: starting layer requested 0 ... (n_lods-1)
        @type input_idx: int
        @param output_idx: ending layer requested 0 ... (n_lods-1)
        @type output_idx: int
        """

    def get_upscale_domain_layer_by_lod(self, input_lod, output_lod) -> tfkl.Layer:
        if output_lod == input_lod:
            return tfkl.Activation('linear')
        input_idx = max((self.n_lods - 1) - output_lod, 0)
        output_idx = max((self.n_lods - 1) - input_lod, 0)
        return self.get_upscale_domain_by_range(input_idx, output_idx)

    def get_conform_to_output_layer_by_lod(self, input_lod) -> tfkl.Layer:
        return self.get_upscale_domain_layer_by_lod(input_lod, self.n_lods)

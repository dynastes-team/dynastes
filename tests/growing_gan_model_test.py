import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.framework import test_util
from tensorflow_core.python.keras.api._v2.keras import layers as tfkl
import timeit
import dynastes as d
from dynastes.models.growing_gan_models import GrowingGanGenerator, GrowingGanClassifier


class Simple2DGrowingGanGenerator(GrowingGanGenerator):

    def __init__(self,
                 n_lods,
                 **kwargs):
        super(Simple2DGrowingGanGenerator, self).__init__(n_lods=n_lods, **kwargs)
        self.gan_layers = [tfkl.UpSampling2D(interpolation='bilinear') for _ in range(n_lods)]
        self.dom_layers = [tfkl.Activation('linear') for _ in range(n_lods + 1)]

    def interpolate_domain(self, x, y, interp):
        return x + (y - x) * interp

    def get_gan_layer_by_lod(self, lod) -> tfkl.Layer:

        if lod > (self.n_lods - 1):
            # Return latent
            return tfkl.Activation('linear')
        return self.gan_layers[(self.n_lods - 1) - lod]

    def get_to_domain_layer_by_lod(self, lod) -> tfkl.Layer:
        return self.dom_layers[(self.n_lods - 1) - lod]

    def get_upscale_domain_layer_by_lod(self, input_lod, output_lod) -> tfkl.Layer:
        if output_lod == input_lod:
            return tfkl.Activation('linear')
        levels = [2] * (output_lod - input_lod)
        if len(levels) == 0:
            return tfkl.Activation('linear')
        scale = np.cumprod(levels)[-1]
        return tfkl.UpSampling2D(size=(scale, scale))

    def get_conform_to_output_layer_by_lod(self, input_lod) -> tfkl.Layer:
        return self.get_upscale_domain_layer_by_lod(input_lod, self.n_lods - 1)


class GrowingGanGeneratorTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            in_z_hw = 2
            n_lods = 4
            z = np.random.random(size=(1, in_z_hw, in_z_hw, 1)).astype(np.float32)
            gen = Simple2DGrowingGanGenerator(n_lods=n_lods)
            y = gen(z, lod_in=2.)
            ex_hw = in_z_hw * (2 ** (n_lods))
            self.assertShapeEqual(np.ones(shape=(1, ex_hw, ex_hw, 1)), y)


class Real2DGrowingGanGenerator(GrowingGanGenerator):

    def __init__(self,
                 n_lods,
                 **kwargs):
        super(Real2DGrowingGanGenerator, self).__init__(n_lods=n_lods, **kwargs)
        self.gan_layers = [tfkl.Conv2DTranspose(2 ** (n_lods - i), kernel_size=(3, 3), strides=(2, 2), padding='same')
                           for i in range(n_lods)]
        self.dom_layers = [tfkl.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same') for _ in
                           range(n_lods + 1)]

    def interpolate_domain(self, x, y, interp):
        return x + (y - x) * interp

    def get_gan_layer_by_lod(self, lod) -> tfkl.Layer:
        if lod > (self.n_lods - 1):
            return tfkl.Activation('linear')
        return self.gan_layers[(self.n_lods - 1) - lod]

    def get_to_domain_layer_by_lod(self, lod) -> tfkl.Layer:
        return self.dom_layers[(self.n_lods - 1) - lod]

    def get_upscale_domain_layer_by_lod(self, input_lod, output_lod) -> tfkl.Layer:
        if output_lod == input_lod:
            return tfkl.Activation('linear')
        levels = [2] * (output_lod - input_lod)
        if len(levels) == 0:
            return tfkl.Activation('linear')
        scale = np.cumprod(levels)[-1]
        return tfkl.UpSampling2D(size=(scale, scale))

    def get_conform_to_output_layer_by_lod(self, input_lod) -> tfkl.Layer:
        return self.get_upscale_domain_layer_by_lod(input_lod, self.n_lods - 1)


class GrowingGanGeneratorTestReal(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            in_z_hw = 2
            n_lods = 2
            z = np.random.random(size=(1, in_z_hw, in_z_hw, 2 ** (n_lods))).astype(np.float32)
            gen = Real2DGrowingGanGenerator(n_lods)
            y = gen(z, lod_in=0.5)
            ex_hw = in_z_hw * (2 ** (n_lods))
            self.assertShapeEqual(np.ones(shape=(1, ex_hw, ex_hw, 3)), y)


class Complex2DGrowingGanGenerator(GrowingGanGenerator):

    def __init__(self,
                 strides=[1, 2, 3],
                 **kwargs):
        n_lods = len(strides)
        super(Complex2DGrowingGanGenerator, self).__init__(n_lods=n_lods, **kwargs)

        self.strides = strides
        channels = [2 ** (n_lods - i) for i in range(n_lods)]
        self.gan_layers = [tfkl.Conv2DTranspose(ch, kernel_size=(3, 3), strides=(s, s), padding='same') for ch, s in
                           zip(channels, self.strides)]
        self.dom_layers = [tfkl.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same') for _ in
                           range(self.n_lods + 1)]

    def interpolate_domain(self, x, y, interp):
        return x + (y - x) * interp

    def get_gan_layer_by_lod(self, lod) -> tfkl.Layer:
        if lod > (self.n_lods - 1):
            return tfkl.Activation('linear')
        return self.gan_layers[(self.n_lods - 1) - lod]

    def get_to_domain_layer_by_lod(self, lod) -> tfkl.Layer:
        return self.dom_layers[(self.n_lods - 1) - lod]

    def get_upscale_domain_layer_by_lod(self, input_lod, output_lod) -> tfkl.Layer:
        if output_lod == input_lod:
            return tfkl.Activation('linear')
        input_idx = max((self.n_lods - 1) - output_lod, 0)
        output_idx = max((self.n_lods - 1) - input_lod, 0)
        strides = (self.strides)[input_idx : output_idx]
        if len(strides) == 0:
            return tfkl.Activation('linear')
        scale = np.cumprod(strides)[-1]
        return tfkl.UpSampling2D(size=(scale, scale))

    def get_conform_to_output_layer_by_lod(self, input_lod) -> tfkl.Layer:
        return self.get_upscale_domain_layer_by_lod(input_lod, self.n_lods)


class GrowingGanGeneratorTestComplex(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        print('GrowingGanGeneratorTestComplex - test_simple')
        with custom_object_scope(d.object_scope):
            in_z_hw = 1
            strides = [7,2,3,1,3,5,1]
            z = np.random.random(size=(1, in_z_hw, in_z_hw, 2**len(strides)+1)).astype(np.float32)
            gen = Complex2DGrowingGanGenerator(strides)
            for i in range(len(strides)*2):
                y = gen(z, lod_in=i / 2)
                ex_hw = in_z_hw * np.cumprod(strides)[-1]
                self.assertShapeEqual(np.ones(shape=(1, ex_hw, ex_hw, 3)), y)
            max_lod = float(len(strides)+1)
            #Test timing
            def lod_0():
                y = gen(z, lod_in=0.)
            def lod_max():
                y = gen(z, lod_in=max_lod)
            time_l0 = timeit.timeit(lod_0, number=2)
            time_l7 = timeit.timeit(lod_max, number=2)
            print('LOD time diff', (time_l7 / time_l0))


class Simple2DGrowingGanClassifier(GrowingGanClassifier):

    def interpolate_domain(self, x, y, interp):
        return x + (y - x) * interp

    def get_gan_layer_by_lod(self, lod) -> tfkl.Layer:
        return tfkl.AveragePooling2D((2, 2), strides=(2, 2))

    def get_from_domain_layer_by_lod(self, lod) -> tfkl.Layer:
        return tfkl.Activation('linear')

    def get_downscale_domain_layer_by_lod(self, input_lod, output_lod) -> tfkl.Layer:
        if output_lod == input_lod:
            return tfkl.Activation('linear')
        levels = [2] * (input_lod - output_lod)
        if len(levels) == 0:
            return tfkl.Activation('linear')
        scale = np.cumprod(levels)[-1]
        return tfkl.AveragePooling2D((scale, scale), strides=(scale, scale))

    def get_input_transform_layer_by_lod(self, lod) -> tfkl.Layer:

        return self.get_downscale_domain_layer_by_lod(self.n_lods, lod)


class GrowingGanClassifierTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            in_hw = 32
            n_lods = 3
            z = np.random.random(size=(1, in_hw, in_hw, 1)).astype(np.float32)
            cls = Simple2DGrowingGanClassifier(n_lods=n_lods)
            y = cls(z, lod_in=1.)
            ex_hw = in_hw // (2 ** (n_lods))
            self.assertShapeEqual(np.ones(shape=(1, ex_hw, ex_hw, 1)), y)


class GrowingEndToEndTest(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            n_lods = 3
            z = np.random.random(size=(1, 64, 48, 1)).astype(np.float32)
            z = tf.convert_to_tensor(z)
            gen = Simple2DGrowingGanGenerator(n_lods=n_lods)
            cls = Simple2DGrowingGanClassifier(n_lods=n_lods)
            _z = z
            for i in range(int(n_lods * 2.)):
                _z = gen(cls(_z, lod_in=i / 2.), lod_in=i / 2.)
            self.assertLess(tf.reduce_mean(tf.abs(z - _z)).numpy(), 0.26)


class Real2DGrowingGanClassifier(GrowingGanClassifier):

    def __init__(self, n_lods=3,
                 **kwargs):
        super(Real2DGrowingGanClassifier, self).__init__(n_lods=n_lods, **kwargs)
        self.gan_layers = [tfkl.Conv2D(2 ** (i + 1), kernel_size=(3, 3), strides=(2, 2), padding='same') for i in
                           range(0, n_lods)]
        self.from_dom_layers = [tfkl.Conv2D(2 ** (i), kernel_size=(1, 1), strides=(1, 1), padding='same') for i in
                                range(0, n_lods + 1)]

    def interpolate_domain(self, x, y, interp):
        return x + (y - x) * interp

    def get_gan_layer_by_lod(self, lod) -> tfkl.Layer:
        if lod < 0:
            raise ArithmeticError
        return self.gan_layers[(self.n_lods - lod)]

    def get_from_domain_layer_by_lod(self, lod) -> tfkl.Layer:
        return self.from_dom_layers[(self.n_lods - lod)]

    def get_downscale_domain_layer_by_lod(self, input_lod, output_lod) -> tfkl.Layer:
        if output_lod == input_lod:
            return tfkl.Activation('linear')
        levels = [2] * (input_lod - output_lod)
        if len(levels) == 0:
            return tfkl.Activation('linear')
        scale = np.cumprod(levels)[-1]
        if scale == 1:
            return tfkl.Activation('linear')
        return tfkl.AveragePooling2D((scale, scale), strides=(scale, scale), padding='same')

    def get_input_transform_layer_by_lod(self, lod) -> tfkl.Layer:
        return self.get_downscale_domain_layer_by_lod(self.n_lods, lod)


class GrowingGanClassifierTestReal(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            in_hw = 32
            n_lods = 4
            z = np.random.random(size=(1, in_hw, in_hw, 1)).astype(np.float32)
            cls = Real2DGrowingGanClassifier(n_lods=n_lods)
            y = cls(z, lod_in=3.)
            ex_hw = in_hw // (2 ** (n_lods))
            self.assertShapeEqual(np.ones(shape=(1, ex_hw, ex_hw, 2 ** n_lods)), y)


class ComplexGanClassifier(GrowingGanClassifier):

    def __init__(self,
                 strides=[2, 3, 4, 2, 5],
                 **kwargs):
        n_lods = len(strides)
        super(ComplexGanClassifier, self).__init__(n_lods=n_lods, **kwargs)
        self.strides = strides
        channels = [2**(i+1) for i in range(len(self.strides)+1)]
        #channels = [3, 4, 4, 8, 16, 32]
        self.gan_layers = []
        self.dom_layers = []
        for i, (s, ch) in enumerate(zip(self.strides, channels[1:])):
            self.gan_layers.append(tfkl.Conv2D(ch, kernel_size=(3, 3), strides=(s, s), padding='same'))
        for ch in channels:
            self.dom_layers.append(tfkl.Conv2D(ch, kernel_size=(1, 1), strides=(1, 1), padding='same'))

    def interpolate_domain(self, x, y, interp):
        return x + (y - x) * interp

    def get_gan_layer_by_lod(self, lod) -> tfkl.Layer:
        if lod < 0:
            raise ArithmeticError
        return self.gan_layers[(self.n_lods - lod)]

    def get_from_domain_layer_by_lod(self, lod) -> tfkl.Layer:
        return self.dom_layers[(self.n_lods - lod)]

    def get_downscale_domain_layer_by_lod(self, input_lod, output_lod) -> tfkl.Layer:
        if output_lod == input_lod:
            return tfkl.Activation('linear')
        strides = self.strides[self.n_lods - input_lod: self.n_lods - output_lod]
        if len(strides) == 0:
            return tfkl.Activation('linear')
        scale = np.cumprod(strides)[-1]
        if scale == 1:
            return tfkl.Activation('linear')
        return tfkl.AveragePooling2D((scale, scale), strides=(scale, scale), padding='same')

    def get_input_transform_layer_by_lod(self, lod) -> tfkl.Layer:
        return self.get_downscale_domain_layer_by_lod(self.n_lods, lod)


class GrowingGanClassifierTestComplex(tf.test.TestCase):
    @test_util.use_deterministic_cudnn
    def test_simple(self):
        with custom_object_scope(d.object_scope):
            strides = [1,3,7,5,3]
            in_base = 1
            in_hw = in_base * np.cumprod(strides)[-1]
            n_lods = len(strides)
            z = np.random.random(size=(1, in_hw, in_hw, 1)).astype(np.float32)
            cls = ComplexGanClassifier(strides)
            y = cls(z, lod_in=3.)
            ex_size_hw = in_hw // np.cumprod(strides)[-1]
            self.assertShapeEqual(np.ones(shape=(1, ex_size_hw, ex_size_hw, 2**(n_lods+1))), y)

# Dynastes
[![Version](https://img.shields.io/pypi/v/dynastes.svg)](https://pypi.org/project/dynastes/)
![License](https://img.shields.io/pypi/l/keras-bert.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)

### Description

Collection of various of my custom TensorFlow-Keras 2.0+ layers, utils and such.

The focus of this library is on time-series, audio, DSP and GAN related networks

### Install

```bash
pip install dynastes
```

### Layers
- Localized Attention (1D and 2D)
  - Perform attention within "kernels", a bit like convolution
- Time-Delay Neural Network Layers

All layers support Spectral Normalization of kernels:
```
kernel_normalizer='spectral'
```
All you need to do in a GAN training is then to call network(x/z, training=True) when training generator or discriminator, updates are automatically performed on the u-variable if training=True. This is enabled by having a "normalizers" dictionary for every weight.
If you implement a custom layer that inherits from DynastesBaseLayer you can assign spectral normalization simply by passing wname_normalizer to the creation args, where wname is the name you give your weight.
This has some caveats, if you call super.get_weight(name) you get the normalized weight, not the actual var / rvar

### Regularizers, Normalization, Constraints, Initializers
- Orthogonal Regularization
- Spectral Normalization

### Functions
- nD-Attention
  - Supports "Multi-Query Attention" from [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)

### Roadmap:
- More attention variants (1D, 2D, Relative, Local, Area) from T2T
- GAN-scaffoldings (ProGAN, StyleGAN, BiGAN, BiStyleGAN?)

### Why?
Keras in TensorFlow 2.0 is nice, but sometimes you need exotic layers and functions that are cumbersome to implement, and I've found myself reimplementing or porting parts of T2T and other things for work and in private, over and over. This library aims to consolidate some of that and maintain tests for it.

### The name "Dynastes"
Dynastes is a genus of large beetles belonging to the subfamily Dynastinae, rhinoceros [ῥῑνόκερως (rhīnókerōs)] beetles and it is also the name of the son of Heracles and Erato (Thespius 49th daughter). This is a play on the word Keras [κέρας (kéras, “horn”)].

#### Notes about sources:

This repository borrows code heavily from:

[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/)

[TensorFlow 2.0 Tutorials](https://www.tensorflow.org/tutorials/)

_Code is copied for stability onto this repository and attribution available when possible_

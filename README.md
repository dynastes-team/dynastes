# Dynastes
[![Version](https://img.shields.io/pypi/v/dynastes.svg)](https://pypi.org/project/dynastes/)
![License](https://img.shields.io/pypi/l/keras-bert.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)

### Description

Collection of various of my custom TensorFlow-Keras 2.0+ layers, utils and such
Focus is on time-series, audio, DSP and GAN related networks

### Install

```bash
pip install dynastes
```

### Layers
Localized Attention (1D and 2D)
* Perform attention within "kernels", a bit like convolution

Time-Delay Neural Network Layers

All layers support Spectral Normalization of kernels:
```
kernel_normalizer='spectral'
```

### Functions
ND-Attention
* Supports "Multi-Query Attention" from [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)

### Roadmap:
* More attention variants (1D, 2D, Relative, Local, Area) from T2T
* GAN-scaffoldings (ProGAN, StyleGAN, BiGAN, BiStyleGAN?)

### Why?
Keras in TensorFlow 2.0 is nice, but sometimes you need exotic layers and functions that are cumbersome to implement, and I've found myself reimplementing or porting parts of T2T and other things for work and in private, over and over. This library aims to consolidate some of that and maintain tests for it.

### The name "Dynastes"
Dynastes is a genus of large beetles belonging to the subfamily Dynastinae, rhinoceros [ῥῑνόκερως (rhīnókerōs)] beetles and it is also the name of the son of Heracles and Erato (Thespius 49th daughter). This is a play on the word Keras [κέρας (kéras, “horn”)].

#### Notes about sources:

This repository borrows code heavily from:

[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/)

[TensorFlow 2.0 Tutorials](https://www.tensorflow.org/tutorials/)

_Code is copied for stability onto this repository and attribution available when possible_

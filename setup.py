import platform

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Dependencies
REQUIRED_PACKAGES = [
    'six >= 1.10.0',
    'tensorflow >= 2.0.0'
]

setuptools.setup(
    name="dynastes",
    version="0.0.1",
    author="Göran Sandström (Veqtor)",
    author_email="v3qt0r@gmail.com",
    description="A collection of layers and utils for TensorFlow (Keras) 2.+",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/veqtor/dynastes",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

import os
import codecs
import re
from setuptools import setup, find_packages

tensorflow_dependency = 'tensorflow'
tensorflow_version = '2.0.0'
current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with codecs.open(os.path.join(current_path, *parts), 'r', 'utf8') as reader:
        return reader.read()


def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


setup(
    name="dynastes",
    version=find_version('dynastes', '__init__.py'),
    author="Göran Sandström (Veqtor)",
    author_email="v3qt0r@gmail.com",
    description="A collection of layers and utils for TensorFlow (Keras) 2.+",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/veqtor/dynastes",
    install_requires=[
        "numpy",
        "six >= 1.10.0",
        '{}>={}'.format(tensorflow_dependency, tensorflow_version),
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

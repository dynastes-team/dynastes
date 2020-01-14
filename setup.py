import os
import codecs
import re
from setuptools import setup, find_packages

current_path = os.path.abspath(os.path.dirname(__file__))

def parse_ver_as_float(ver):
    return float('.'.join(ver.split('.')[:2]))

def get_requirements_dynamic():
    requires = []
    try:
        import tensorflow
        if parse_ver_as_float(tensorflow.__version__) < 2.1:
            raise ImportError
    except:
        requires.append('tensorflow>=2.1.0')
    try:
        import tensorflow_probability
        if parse_ver_as_float(tensorflow_probability.__version__) < 0.8:
            raise ImportError
    except:
        print('Warning! TensorFlow probability is missing!')
    try:
        import tensorflow_addons
        if parse_ver_as_float(tensorflow_addons.__version__) < 0.7:
            raise ImportError
    except:
        print('Warning! TensorFlow addons are missing!')
    requires.append('numpy')
    requires.append('six >= 1.10.0')

def read_file(*parts):
    with codecs.open(os.path.join(current_path, *parts), 'r', 'utf8') as reader:
        return reader.read()

def get_requirements(*parts):
    with codecs.open(os.path.join(current_path, *parts), 'r', 'utf8') as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))

def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


setup(
    name="dynastes",
    version=find_version('dynastes', '__init__.py'),
    author="Team Dynastes",
    author_email="v3qt0r@gmail.com",
    description="A collection of layers and utils for TensorFlow (Keras) 2.+",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/dynastes-team/dynastes",
    install_requires=get_requirements_dynamic(),
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

# Copyright 2020 AWS Neuron. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
import setuptools
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel as BDistWheelCommandBase


class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True


class InstallCommand(InstallCommandBase):
    """Override the installation dir."""

    def finalize_options(self):
        ret = InstallCommandBase.finalize_options(self)
        self.install_lib = self.install_platlib
        return ret


class BDistWheelCommand(BDistWheelCommandBase):

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = super().get_tag()
        python, abi = 'py3', 'none'
        return python, abi, plat


if sys.platform == 'linux':
    PLUGIN_NAME = 'libaws_neuron_plugin.so'
elif sys.platform == 'darwin':
    PLUGIN_NAME = 'libaws_neuron_plugin.dylib'
else:
    raise NotImplementedError('platform {} is unsupported'.format(sys.platform))


setuptools.setup(
    name='tensorflow-neuron',
    version='_VERSION',
    description='TensorFlow Neuron integration',
    author='AWS Neuron SDK',
    author_email='aws-neuron-support@amazon.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow aws neuron',
    include_package_data=True,
    packages=setuptools.PEP420PackageFinder.find(),
    package_data={
        'tensorflow-plugins': [PLUGIN_NAME],
        'tensorflow_neuron': ['../tensorflow.py'],
    },
    distclass=BinaryDistribution,
    cmdclass={
        'bdist_wheel': BDistWheelCommand,
        'install': InstallCommand,
    },
    install_requires=[
        'tensorflow ~= 1.15.0',
        'tensorboard-neuron ~= 1.15.0',
    ],
)

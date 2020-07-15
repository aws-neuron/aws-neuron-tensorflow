import setuptools
from wheel.bdist_wheel import bdist_wheel


class tfn_bdist_wheel(bdist_wheel):

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = super().get_tag()
        python, abi = 'py3', 'none'
        return python, abi, plat


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
        'tensorflow-plugins': ['aws_neuron_plugin.so'],
    },
    cmdclass={'bdist_wheel': tfn_bdist_wheel},
    install_requires=[
        'tensorflow ~= 1.15.0',
        'tensorboard-neuron ~= 1.15.0',
    ],
)

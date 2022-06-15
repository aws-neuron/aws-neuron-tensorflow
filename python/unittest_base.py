# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
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
import os
import unittest
from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.eager import function


class TestV1Only(tf.test.TestCase, metaclass=RemoveTestSession):

    @classmethod
    def setUpClass(cls):
        if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
            raise unittest.SkipTest('tf v1 only tests is not supported under tf 2.x')


class RemoveTestSession(type):

    def __new__(mcs, name, bases, dct):
        try:
            del tf.test.TestCase.test_session
        except AttributeError:
            pass
        return type.__new__(mcs, name, bases, dct)


class TestV2Only(tf.test.TestCase, metaclass=RemoveTestSession):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
            raise unittest.SkipTest('tf v2 only tests is not supported under tf 1.x')

    def setUp(self):
        super().setUp()
        tf.random.set_seed(15213)

        def fake_call(*args, **kwargs):
            raise unittest.SkipTest('skipping inference in compile-only mode')

        if 'NEURON_TF_COMPILE_ONLY' in os.environ:
            function.ConcreteFunction.__call__ = fake_call


def xfail_for_versions(*versions):
    def major_minor(ver):
        return LooseVersion(ver).version[:2]
    def wrapper(test_func):
        if any(major_minor(tf.__version__) == major_minor(ver) for ver in versions):
            test_func = unittest.expectedFailure(test_func)
        return test_func
    return wrapper

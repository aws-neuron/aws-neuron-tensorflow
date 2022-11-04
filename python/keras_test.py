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

import shutil
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras.applications.resnet50 import ResNet50
import tensorflow.neuron as tfn
from tensorflow_neuron.python.unittest_base import TestV1Only
from tensorflow_neuron.python.graph_util_test import _assert_compiler_success


_RANDOM_SEED = 15213


class TestKeras(TestV1Only):

    def test_keras_resnet50_float16_compile(self):
        # Instantiate Keras ResNet50 model
        keras.backend.set_learning_phase(0)
        keras.backend.set_floatx('float16')
        model = ResNet50(weights='imagenet')

        # Export SavedModel
        model_dir = './keras_resnet50_float16'
        shutil.rmtree(model_dir, ignore_errors=True)
        tf.saved_model.simple_save(keras.backend.get_session(), model_dir,
                                   {'input': model.inputs[0]}, {'output': model.outputs[0]})

        # Compile using Neuron
        compiled_model_dir = './keras_resnet50_float16_neuron'
        shutil.rmtree(compiled_model_dir, ignore_errors=True)
        compile_output = tfn.saved_model.compile(
            model_dir, compiled_model_dir,
            model_feed_dict={'input' : np.random.rand(1, 224, 224, 3)})
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ['serve'], compiled_model_dir)
            _assert_compiler_success(sess.graph)

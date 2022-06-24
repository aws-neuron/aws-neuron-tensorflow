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

import tensorflow as tf
import tensorflow.neuron as tfn
import unittest
import os
from tensorflow_neuron.python.unittest_base import TestV2Only
from contextlib import contextmanager

class TestTraceReduceNeffSize(TestV2Only):
    def setUp(self):
        # returns None if NEURON_CC_FLAGS is not set
        self.old_flags = os.environ.get('NEURON_CC_FLAGS')
        os.environ['NEURON_CC_FLAGS'] = "--extract-weights"
    
    def tearDown(self):
        if self.old_flags is not None:
            os.environ['NEURON_CC_FLAGS'] = self.old_flags
        else:
            # NEURON_CC_FLAGS variable never existed in the first place
            # so it needs to be deleted
            del os.environ['NEURON_CC_FLAGS']

    def test_reduce_neff_size_simple(self):

        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        inputs = [input0]
        outputs = [dense0]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        input0_tensor = tf.random.uniform([1, 3])
        
        model_neuron = tfn.trace(model, input0_tensor)
        os.environ['NEURON_CC_FLAGS']=""
        model_neuron_NO_REDUCED_NEFF_SIZE = tfn.trace(model, input0_tensor)

        #make sure extract weights code was actually triggered
        assert model_neuron._ordered_weights is not None
        assert model_neuron_NO_REDUCED_NEFF_SIZE._ordered_weights is None

        model_dir = os.path.join(self.get_temp_dir(), 'removed_constants_dense')

        model_neuron.save(model_dir)
        model_neuron_reloaded = tf.keras.models.load_model(model_dir)

        res_ref = model(input0_tensor)
        res_neuron_ref = model_neuron_NO_REDUCED_NEFF_SIZE(input0_tensor) 
        res_neuron = model_neuron(input0_tensor)
        res_neuron_reloaded = model_neuron_reloaded(input0_tensor)

        #run it twice to test the cached weights
        res_ref = model(input0_tensor)
        res_neuron_ref = model_neuron_NO_REDUCED_NEFF_SIZE(input0_tensor) 
        res_neuron = model_neuron(input0_tensor)
        res_neuron_reloaded = model_neuron_reloaded(input0_tensor)

        #fails on 1e-4
        self.assertAllClose(res_ref, res_neuron_ref, rtol=1e-3, atol=1e-3)
        self.assertAllClose(res_ref, res_neuron, rtol=1e-3, atol=1e-3)
        self.assertAllClose(res_ref, res_neuron_reloaded, rtol=1e-3, atol=1e-3)


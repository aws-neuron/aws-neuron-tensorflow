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
import argparse
import os
import unittest
from unittest import mock
import numpy as np
import tensorflow as tf
import tensorflow.neuron as tfn
from tensorflow.python import saved_model
from tensorflow_neuron.python.unittest_base import TestV2Only
from tensorflow_neuron.python.utils import _assert_compiler_success_func

tNeuronOp = "NeuronOp"


class TestAutoMulticoreAPI(TestV2Only):
    def test_trace(self):
        """
        Test to see if the neuron op has its attribute correctly modified
        when model is traced via trace call
        """
        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        inputs = [input0]
        outputs = [dense0]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        input0_tensor = tf.random.uniform([1, 3])
        model_neuron = tfn.trace(model, input0_tensor)
        _assert_compiler_success_func(model_neuron.aws_neuron_function)

        num_cores = 4
        multicore_model = tfn.auto_multicore(
            model_neuron, input0_tensor, num_cores=num_cores
        )

        graph_def = multicore_model.aws_neuron_function.graph.as_graph_def()
        for node in graph_def.node:
            if node.op == "NeuronOp":
                auto_multicore_flag = node.attr["_automatic_multicore"].i
                assert auto_multicore_flag == num_cores

    def test_save_load_model(self):
        """
        Test to see if the neuron op has its attribute correctly modified
        when the model has been traced, saved and loaded again
        """
        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        inputs = [input0]
        outputs = [dense0]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        input0_tensor = tf.random.uniform([1, 3])
        model_neuron = tfn.trace(model, input0_tensor)

        num_cores = 4

        model_dir = os.path.join(
            self.get_temp_dir(), "neuron_keras_model_1in_1out_save"
        )
        model_neuron.save(model_dir)

        reload_model = saved_model.load(model_dir)
        multicore_model = tfn.auto_multicore(
            reload_model, input0_tensor, num_cores=num_cores
        )
        graph_def = multicore_model.aws_neuron_function.graph.as_graph_def()
        for node in graph_def.node:
            if node.op == "NeuronOp":
                auto_multicore_flag = node.attr["_automatic_multicore"].i
                assert auto_multicore_flag == num_cores


if __name__ == "__main__":
    unittest.main()

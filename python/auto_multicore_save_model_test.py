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
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.test import get_temp_dir
import tensorflow.neuron as tfn
from tensorflow.python import saved_model
from tensorflow.neuron.python.unittest_base import TestV1Only, TestV2Only
from tensorflow.neuron.python.utils import _assert_compiler_success_func
from tensorflow.neuron.python.auto_multicore_save_model import add_attr_to_model
from tensorflow.neuron.python.graph_util import tag_multicore
from tensorflow.neuron.python.graph_util_test import _assert_compiler_success


class TestAutoMulticoreV1(TestV1Only):
    def test_simple_cli_v1(self):
        """
        Test to see if the neuron op has its attribute correctly modified
        """

        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        inputs = [input0]
        outputs = [dense0]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model_dir = os.path.join(get_temp_dir(), "neuron_keras_model_1in_1out_save")

        tf.saved_model.simple_save(
            keras.backend.get_session(),
            model_dir,
            {"input": model.inputs[0]},
            {"output": model.outputs[0]},
        )

        compiled_model_dir = os.path.join(get_temp_dir(), "compiled_dir")
        compile_output = tfn.saved_model.compile(
            model_dir,
            compiled_model_dir,
            model_feed_dict={"input": np.random.rand(1, 3)},
        )

        num_cores = 4
        new_model_dir = os.path.join(get_temp_dir(), "new_model_dir")
        func_args = [
            compiled_model_dir,
            "--new_model_dir",
            new_model_dir,
            "--num_cores",
            str(num_cores),
        ]
        add_attr_to_model(func_args)

        pred_neuron = tf.contrib.predictor.from_saved_model(new_model_dir)
        op_list = [
            op for op in pred_neuron.graph.get_operations() if op.type == "NeuronOp"
        ]

        assert len(op_list) >= 1
        for op in op_list:
            if op.node_def.attr["_automatic_multicore"].i:
                auto_multicore_flag = op.node_def.attr["_automatic_multicore"].i
                assert auto_multicore_flag == num_cores


class TestAutoMulticoreV2(TestV2Only):
    """Tests for Automatic Replication Multicore API

    We test basic Keras models and also invalid input. Note that the invalid input
    is the only test currently that evaluates if the C++ runtime is working for this API.
    """

    def test_simple_cli(self):
        """
        Test to see if the neuron op has its attribute correctly modified
        """
        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        inputs = [input0]
        outputs = [dense0]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        input0_tensor = tf.random.uniform([1, 3])
        model_neuron = tfn.trace(model, input0_tensor)
        model_dir = os.path.join(
            self.get_temp_dir(), "neuron_keras_model_1in_1out_save"
        )
        model_neuron.save(model_dir)

        new_model_dir = os.path.join(self.get_temp_dir(), "new_model_dir")
        num_cores = 4

        # Simulate the command line interface
        func_args = [
            model_dir,
            "--new_model_dir",
            new_model_dir,
            "--num_cores",
            str(num_cores),
        ]
        add_attr_to_model(func_args)
        converted_model_neuron = saved_model.load(new_model_dir)
        graph_def = converted_model_neuron.aws_neuron_function.graph.as_graph_def()
        for node in graph_def.node:
            if node.op == "NeuronOp":
                auto_multicore_flag = node.attr["_automatic_multicore"].i
                assert auto_multicore_flag == num_cores

    def test_negative_cores_cli(self):
        """
        Test to see requested negative cores will fail properly in runtime
        """
        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        inputs = [input0]
        outputs = [dense0]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        input0_tensor = tf.random.uniform([1, 3])

        model_neuron = tfn.trace(model, input0_tensor)
        model_dir = os.path.join(
            self.get_temp_dir(), "neuron_keras_model_1in_1out_save"
        )
        model_neuron.save(model_dir)

        new_model_dir = os.path.join(self.get_temp_dir(), "new_model_dir")
        num_cores = -1

        # Simulate the command line interface
        func_args = [
            model_dir,
            "--new_model_dir",
            new_model_dir,
            "--num_cores",
            str(num_cores),
        ]
        add_attr_to_model(func_args)
        converted_model_neuron = tf.keras.models.load_model(new_model_dir)

        graph_def = converted_model_neuron.aws_neuron_function.graph.as_graph_def()
        for node in graph_def.node:
            if node.op == "NeuronOp":
                auto_multicore_flag = node.attr["_automatic_multicore"].i
                assert auto_multicore_flag == num_cores

        with self.assertRaises(tf.errors.InvalidArgumentError) as cm:
            converted_model_neuron(input0_tensor)

        self.assertContainsExactSubsequence(
            cm.exception.message, "Num of cores for auto multicore"
        )

    def test_simple_python_api(self):
        """
        Test to see if the neuron op has its attribute correctly modified
        """
        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        inputs = [input0]
        outputs = [dense0]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        input0_tensor = tf.random.uniform([1, 3])
        model_neuron = tfn.trace(model, input0_tensor)

        # Test the python api
        num_cores = 4
        auto_model_neuron = tfn.auto_multicore(
            model_neuron, input0_tensor, num_cores=num_cores
        )

        graph_def = auto_model_neuron.aws_neuron_function.graph.as_graph_def()
        for node in graph_def.node:
            if node.op == "NeuronOp":
                auto_multicore_flag = node.attr["_automatic_multicore"].i
                assert auto_multicore_flag == num_cores

    def test_negative_cores_python_api(self):
        """
        Test to see requested negative cores will fail properly in runtime
        """
        input0 = tf.keras.layers.Input(3)
        dense0 = tf.keras.layers.Dense(3)(input0)
        inputs = [input0]
        outputs = [dense0]
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        input0_tensor = tf.random.uniform([1, 3])
        model_neuron = tfn.trace(model, input0_tensor)

        # Test the python api
        num_cores = -1
        auto_model_neuron = tfn.auto_multicore(
            model_neuron, input0_tensor, num_cores=num_cores
        )

        graph_def = auto_model_neuron.aws_neuron_function.graph.as_graph_def()
        for node in graph_def.node:
            if node.op == "NeuronOp":
                auto_multicore_flag = node.attr["_automatic_multicore"].i
                assert auto_multicore_flag == num_cores

        with self.assertRaises(tf.errors.InvalidArgumentError) as cm:
            auto_model_neuron(input0_tensor)

        self.assertContainsExactSubsequence(
            cm.exception.message, "Num of cores for auto multicore"
        )


if __name__ == "__main__":
    unittest.main()

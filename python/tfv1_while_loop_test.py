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
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.neuron.python import graph_util
from tensorflow.neuron.python.unittest_base import TestV1Only


_RANDOM_SEED = 15213


class TestInferenceGraphFromSession(TestV1Only):

    def test_while_loop(self):
        np.random.seed(_RANDOM_SEED)
        kernel0_np = np.random.uniform(-0.1, 0.1, size=[128, 128]).astype(np.float16)
        maximum_iterations = 5
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, shape=[128, 128], name='input0')

            def body(input0):
                kernel0 = tf.constant(kernel0_np, name='kernel0')
                matmul0 = tf.matmul(input0, kernel0, transpose_a=True, name='matmul0')
                output0 = tf.nn.relu(matmul0, name='output0')
                return output0

            output0 = tf.while_loop(lambda x: True, body, [input0], maximum_iterations=maximum_iterations)
            input0_np = np.random.uniform(-0.1, 0.1, size=[128, 128]).astype(np.float16)
            feed_dict = {input0.name: input0_np}
            result_np = input0_np
            for _ in range(maximum_iterations):
                result_np = np.matmul(result_np.T, kernel0_np)
                result_np = np.maximum(result_np, 0.0)
            result_tf = sess.run(output0, feed_dict)
            np.testing.assert_allclose(result_np, result_tf, rtol=1e-2, atol=1e-4)
            infer_graph = graph_util.inference_graph_from_session(
                sess, compiler_workdir='./workdir', output_tensors={output0})
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run(output0.name, feed_dict=feed_dict)
                np.testing.assert_allclose(result_neuron, result_tf, rtol=1e-2, atol=1e-4)

    def test_while_parloop(self):
        np.random.seed(_RANDOM_SEED)
        kernel0_np = np.random.uniform(-0.1, 0.1, size=[8, 8]).astype(np.float16)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, shape=[8, 8], name='input0')

            def body(input1):
                kernel0 = tf.constant(kernel0_np, name='kernel0')
                matmul0 = tf.matmul(input0, kernel0, name='matmul0')
                output0 = tf.nn.relu(matmul0, name='output0')
                return output0

            output0 = tf.while_loop(lambda x: True, body, [input0], maximum_iterations=6,
                                    parallel_iterations=16, name='while')
            input0_np = np.random.uniform(-1.0, 1.0, size=[8, 8]).astype(np.float16)
            feed_dict = {input0.name: input0_np}
            result_tf_list = [sess.run(output0, feed_dict) for _ in range(2)]
            infer_graph = graph_util.inference_graph_from_session(
                sess, compiler_workdir='./workdir', output_tensors={output0},
                no_fuse_ops={'while/kernel0'})
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron_list = [sess.run(output0.name, feed_dict) for _ in range(2)]
            for result_neuron, result_tf in zip(result_neuron_list, result_tf_list):
                np.testing.assert_allclose(result_neuron, result_tf, rtol=1e-2, atol=1e-5)


class TestShapeInference(TestV1Only):

    def test_with_inputs_while_loop(self):
        np.random.seed(_RANDOM_SEED)
        kernel0_np = np.random.uniform(-0.1, 0.1, size=[128, 128]).astype(np.float16)
        maximum_iterations = 5
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, shape=[128, 128], name='input0')

            def body(input0):
                kernel0 = tf.constant(kernel0_np, name='kernel0')
                matmul0 = tf.matmul(input0, kernel0, transpose_a=True, name='matmul0')
                output0 = tf.nn.relu(matmul0, name='output0')
                return output0

            output0 = tf.while_loop(lambda x: True, body, [input0], maximum_iterations=maximum_iterations)
            input0_np = np.random.uniform(-0.1, 0.1, size=[128, 128]).astype(np.float16)
            feed_dict = {input0.name: input0_np}
            result_np = input0_np
            for _ in range(maximum_iterations):
                result_np = np.matmul(result_np.T, kernel0_np)
                result_np = np.maximum(result_np, 0.0)
            result_tf = sess.run(output0, feed_dict)
            np.testing.assert_allclose(result_np, result_tf, rtol=1e-2, atol=1e-4)
            infer_graph = graph_util.inference_graph_from_session(
                sess, compiler_workdir='./workdir', output_tensors={output0}, feed_dict=feed_dict)
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run(output0.name, feed_dict=feed_dict)
                np.testing.assert_allclose(result_neuron, result_tf, rtol=1e-2, atol=1e-4)


def _assert_neuron_op(infer_graph):
    op_list = [op for op in infer_graph.get_operations() if op.type == 'NeuronOp']
    if not op_list:
        raise AssertionError('No NeuronOp is found')
    return op_list


def _assert_compiler_success(infer_graph):
    op_list = _assert_neuron_op(infer_graph)
    for op in op_list:
        if not op.node_def.attr['executable'].s:
            raise AssertionError('NeuronOp {} is not compiled'.format(op.name))


if __name__ == '__main__':
    unittest.main()

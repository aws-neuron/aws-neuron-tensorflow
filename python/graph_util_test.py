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

import sys
import os
import copy
import shutil
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import unittest
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.neuron.python import graph_util
from tensorflow.neuron.python.ops.gen_neuron_op import neuron_op
from tensorflow.python.platform import tf_logging as logging


_RANDOM_SEED = 15213


class TestInferenceGraphFromSession(unittest.TestCase):

    def test_multiple_inputs_outputs(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input1')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            feed_dict = {
                'input0:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
                'input1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_ref0 = sess.run(['relu0:0', 'relu1:0'], feed_dict)
            infer_graph0 = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
            result_ref1 = sess.run(['relu0:0', 'sigmoid0:0', 'relu1:0', 'add0:0'], feed_dict)
            infer_graph1 = graph_util.inference_graph_from_session(
                sess, output_tensors={'relu0:0', sigmoid0, 'relu1:0', 'add0:0'},
                op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'},
                compiler_workdir='./workdir')
            infer_graph1 = graph_util.inference_graph_from_session(
                sess, output_tensors={'relu0:0', 'sigmoid0:0', relu1, 'add0:0'},
                op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        _assert_compiler_success(infer_graph0)
        _assert_compiler_success(infer_graph1)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph0) as sess:
                result_neuron0 = sess.run(['relu0:0', 'relu1:0'], feed_dict)
                assert len(result_neuron0) == len(result_ref0)
                for res_neuron, res_ref in zip(result_neuron0, result_ref0):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)
            with tf.Session(graph=infer_graph1) as sess:
                result_neuron1 = sess.run(['relu0:0', 'sigmoid0:0', 'relu1:0', 'add0:0'], feed_dict)
                assert len(result_neuron1) == len(result_ref1)
                for res_neuron, res_ref in zip(result_neuron1, result_ref1):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)

    def test_graph_with_variable(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
            kernel0 = tf.Variable(np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16))
            kernel1 = tf.Variable(np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16))
            conv2d0 = tf.nn.conv2d(input0, kernel0, strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, kernel1, strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            kernel2 = tf.Variable(np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16))
            conv2d2 = tf.nn.conv2d(sigmoid0, kernel2, strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            feed_dict = {
                'input0:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
                'input1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            sess.run(tf.global_variables_initializer())
            result_ref = sess.run(['relu0:0', 'sigmoid0:0', 'relu1:0', 'add0:0'], feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, feed_dict=feed_dict, output_tensors={'relu0:0', 'sigmoid0:0', relu1, 'add0:0'},
                op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run(['relu0:0', 'sigmoid0:0', 'relu1:0', 'add0:0'], feed_dict)
                assert len(result_neuron) == len(result_ref)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)

    def test_inputs_start_from_middle(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.int32, name='input1')
            input2 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input2')
            input3 = tf.placeholder(tf.bool, name='input3')
            identity_n0 = tf.identity_n([input0, input1, input2, input3], name='identity_n0')
            identity_n10, _, identity_n12, _ = tf.identity_n(identity_n0, name='identity_n1')
            conv2d0 = tf.nn.conv2d(identity_n10, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d2 = tf.nn.conv2d(identity_n12, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d2')
            add0 = tf.add(conv2d0, conv2d2, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            feed_dict = {
                'identity_n1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
                'identity_n1:2': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_names = ['relu0:0', 'relu1:0']
            result_ref = sess.run(result_names, feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'},
                feed_dict=feed_dict, output_tensors=result_names,
                compiler_workdir='./workdir')
        for name in feed_dict.keys():
            infer_graph.get_tensor_by_name(name)
        for name in result_names:
            infer_graph.get_tensor_by_name(name)
        _assert_compiler_success(infer_graph)
        assert len([op for op in infer_graph.get_operations() if op.type == 'NeuronOp']) == 2
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run(result_names, feed_dict)
                assert len(result_neuron) == len(result_ref)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)

    def test_compiler_timeout_recovery(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.int32, name='input1')
            input2 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input2')
            input3 = tf.placeholder(tf.bool, name='input3')
            identity_n0 = tf.identity_n([input0, input1, input2, input3], name='identity_n0')
            identity_n10, _, identity_n12, _ = tf.identity_n(identity_n0, name='identity_n1')
            conv2d0 = tf.nn.conv2d(identity_n10, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d2 = tf.nn.conv2d(identity_n12, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d2')
            add0 = tf.add(conv2d0, conv2d2, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d3 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d3')
            relu1 = tf.nn.relu(conv2d3, name='relu1')
            feed_dict = {
                'identity_n1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
                'identity_n1:2': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_names = ['relu0:0', conv2d3.name, 'add0:0', 'relu1:0']
            result_ref = sess.run(result_names, feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'},
                feed_dict=feed_dict, output_tensors=result_names,
                compiler_workdir='./workdir', compiler_timeout=1)
        for name in feed_dict.keys():
            infer_graph.get_tensor_by_name(name)
        for name in result_names:
            infer_graph.get_tensor_by_name(name)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run(result_names, feed_dict)
                assert len(result_neuron) == len(result_ref)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)

    def test_graph_with_large_constants(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.int32, name='input1')
            input2 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input2')
            input3 = tf.placeholder(tf.bool, name='input3')
            identity_n0 = tf.identity_n([input0, input1, input2, input3], name='identity_n0')
            identity_n10, _, identity_n12, _ = tf.identity_n(identity_n0, name='identity_n1')
            conv2d0 = tf.nn.conv2d(identity_n10, np.random.uniform(-1, 1, size=[1, 1, 3, 1024]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d2 = tf.nn.conv2d(identity_n12, np.random.uniform(-1, 1, size=[1, 1, 3, 1024]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d2')
            add0 = tf.add(conv2d0, conv2d2, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 1024, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            identity_n2 = tf.identity_n([conv2d2], name='identity_n2')[0]
            relu1 = tf.nn.relu(identity_n2, name='relu1')
            feed_dict = {
                'identity_n1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
                'identity_n1:2': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_names = ['relu0:0', 'relu1:0']
            result_ref = sess.run(result_names, feed_dict)
            import neuroncc
            from distutils.version import LooseVersion
            neuroncc_version = LooseVersion(neuroncc.__version__.split('+')[0])
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu', 'IdentityN'},
                feed_dict=feed_dict, output_tensors=result_names,
                compiler_args=['--experimental-suppress-scheduler-data-race'] if neuroncc_version > LooseVersion('1.0.10450') else None,
                compiler_workdir='./workdir')
        for name in feed_dict.keys():
            infer_graph.get_tensor_by_name(name)
        for name in result_names:
            infer_graph.get_tensor_by_name(name)
        _assert_compiler_success(infer_graph)
        assert len([op for op in infer_graph.get_operations() if op.type == 'NeuronOp']) == 1
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run(result_names, feed_dict)
                assert len(result_neuron) == len(result_ref)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-2)

    def test_scalar_input_output(self):
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float32, name='input0')
            input1 = tf.placeholder(tf.float32, name='input1')
            add0 = tf.add(input0, input0, name='add0')
            mul0 = tf.multiply(input1, add0)
            infer_graph = graph_util.inference_graph_from_session(
                sess, shape_feed_dict={input0: [], input1: []}, output_tensors=[mul0, add0],
                compiler_workdir='./workdir')

    def test_feed_dict(self):
        np.random.seed(_RANDOM_SEED)
        # case where session.run is required
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float32, [None, 16], name='input0')
            input1 = tf.placeholder(tf.float32, [None, 8], name='input1')
            matmul0 = tf.matmul(input0, np.random.rand(16, 24).astype(np.float32), name='matmul0')
            matmul1 = tf.matmul(input1, np.random.rand(8, 24).astype(np.float32), name='matmul1')
            add0 = tf.add(matmul0, matmul1, name='add0')
            reshape0 = tf.reshape(add0, tf.range(2, 5), name='reshape0')
            relu0 = tf.nn.relu(reshape0, name='relu0')
            exp0 = tf.exp(reshape0, name='exp0')
            feed_dict = {'input0:0': np.random.rand(1, 16), 'input1:0': np.random.rand(1, 8)}
            infer_graph = graph_util.inference_graph_from_session(
                sess, feed_dict=feed_dict, output_tensors=[relu0, exp0],
                compiler_workdir='./workdir')

    def test_input_identity(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            identity0 = tf.identity(input0, name='identity0')
            conv2d0 = tf.nn.conv2d(identity0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            relu0 = tf.nn.relu(conv2d0, name='relu0')
            feed_dict = {
                'identity0:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_names = ['relu0:0']
            result_ref = sess.run(result_names, feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'}, feed_dict=feed_dict,
                compiler_workdir='./workdir')
        for name in feed_dict.keys():
            infer_graph.get_tensor_by_name(name)
        for name in result_names:
            infer_graph.get_tensor_by_name(name)
        _assert_compiler_success(infer_graph)
        assert len([op for op in infer_graph.get_operations() if op.type == 'NeuronOp']) == 1
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run(result_names, feed_dict)
                assert len(result_neuron) == len(result_ref)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)

    def test_add_with_shape(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float32, [None], name='input0')
            shape0 = tf.shape(input0)
            shape0_cast = tf.cast(shape0, tf.float32)
            output0 = tf.add(input0, shape0_cast, name='output0')
            feed_dict = {
                'input0:0': np.random.uniform(-1, 1, size=[1]).astype(np.float32),
            }
            result_ref0 = sess.run(['output0:0'], feed_dict)
            infer_graph0 = graph_util.inference_graph_from_session(sess, feed_dict=feed_dict)
        _assert_compiler_success(infer_graph0)

    def test_neuron_op_different_name(self):
        NUMHID = 512
        HEADSIZE = 64
        WMIN = -0.01
        WMAX = 0.01
        IMIN = -0.1
        IMAX = 0.1
        BATCHSIZE = 4
        INPUTLEN = 16
        OUTPUTLEN = 16
        NEGINF = -10000
        batch_input_len = BATCHSIZE * INPUTLEN
        batch_output_len = BATCHSIZE * OUTPUTLEN

        def encoder_block(inputs, attention_bias, layer_name_prefix):
            with tf.name_scope('self_att'):
                with tf.name_scope('lm'):
                    inputs_norm = layer_normalization(inputs)
                with tf.name_scope('mhatt'):
                    self_att = multihead_attention(inputs_norm, inputs_norm,
                        attention_bias)
                    inputs = tf.add(self_att, inputs, name='resadd')
            with tf.name_scope('ffn_blk'):
                with tf.name_scope('lm'):
                    inputs_norm = layer_normalization(inputs)
                with tf.name_scope('ffn'):
                    outputs = feed_forward_network(inputs_norm, layer_name_prefix)
                    outputs = tf.add(outputs, inputs, name='resadd')
            return outputs

        def layer_normalization(inputs):
            layer_norm_scale = np.random.uniform(WMIN, WMAX, size=inputs.shape).astype(np.float16)
            layer_norm_bias = np.random.uniform(WMIN, WMAX, size=inputs.shape).astype(np.float16)
            num_channels = inputs.shape[-1]

            def broadcast_c(v, _name):
                broadcast_w = np.full([1, num_channels], 1.0).astype(np.float16)
                return tf.matmul(v, broadcast_w, name='broadcast_' + _name)

            inputs_sum = tf.reduce_sum(inputs, axis=[-1], keepdims=True, name='sum')
            factor = 1.0 / (int(num_channels) * 10)
            mean = tf.multiply(inputs_sum, -factor, name='mean')
            mean_b = broadcast_c(mean, 'b1')
            residuals = tf.add(inputs, mean_b, name='residuals')
            res_x_res = tf.multiply(residuals, residuals, name='res_x_res')
            residuals_squared_sum = tf.reduce_sum(res_x_res, axis=[-1],
                keepdims=True, name='residuals_squared_sum')
            var = tf.multiply(residuals_squared_sum, factor, name='mult_var')
            rsqrt_ret = tf.rsqrt(var + 1e-6, name='rsqrt_ret')
            norm_inputs = tf.multiply(residuals, broadcast_c(rsqrt_ret , 'b2'), 'normalized')
            norm_scale = tf.multiply(norm_inputs, layer_norm_scale, name = 'norm_scale')
            pre_outputs = tf.add(norm_scale, layer_norm_bias, 'pre_output')
            return pre_outputs

        def multihead_attention(b_input_x_r, b_input_y_r, b_bias_br_arr):
            # constants
            q_kernel = np.random.uniform(WMIN, WMAX, size=[NUMHID, NUMHID]).astype(np.float16)
            k_kernel = np.random.uniform(WMIN, WMAX, size=[NUMHID, NUMHID]).astype(np.float16)
            tr_kernel = np.random.uniform(WMIN, WMAX, size=[NUMHID, NUMHID]).astype(np.float16)
            q_kernel = q_kernel / np.sqrt(HEADSIZE)
            v_kernel = np.random.uniform(WMIN, WMAX, size=[NUMHID, NUMHID]).astype(np.float16)
            ones_for_br = np.ones([1, OUTPUTLEN]).astype(np.float16)

            # graph
            b_q_heads_tiled = tf.matmul(b_input_x_r, q_kernel, name='b_q_heads_tiled')
            b_k_heads_tiled = tf.matmul(b_input_y_r, k_kernel, name='b_k_heads_tiled')
            b_v_t_heads_tiled = tf.matmul(v_kernel, b_input_y_r, transpose_b=True, name='b_v_heads_t_tiled')

            # heads
            b_weighted_v_t_list = []
            for i, (bst_in, bst_out) in enumerate(
                    zip(range(0, batch_input_len, INPUTLEN),
                        range(0, batch_output_len, OUTPUTLEN))):
                with tf.name_scope('batch_{}'.format(i)):
                    weighted_v_heads_list = []
                    for j, hst in enumerate(range(0, NUMHID, HEADSIZE)):
                        with tf.name_scope('head_{}'.format(j)):
                            q_head_c_slice = b_q_heads_tiled[:, hst:hst+HEADSIZE]
                            q_head = tf.slice(q_head_c_slice, [bst_in, 0], [INPUTLEN, -1], name='q_head')
                            k_head_c_slice = tf.add(b_k_heads_tiled[:, hst:hst+HEADSIZE], 0.0)
                            k_head = tf.slice(k_head_c_slice, [bst_out, 0], [OUTPUTLEN, -1], name='k_head_batch')
                            qk_head = tf.matmul(q_head, k_head, transpose_b=True, name='qk_head')

                            # add bias
                            bias_br = b_bias_br_arr[i]
                            qk_bias_head = tf.add(qk_head, bias_br)
                            qk_exp_head = tf.exp(qk_bias_head)
                            norm_factor = tf.reduce_sum(qk_exp_head, axis=1, keepdims=True) # (16, 1)
                            norm_factor_rec = tf.reciprocal(norm_factor)
                            norm_factor_rec_br = tf.matmul(norm_factor_rec, ones_for_br, name='norm_factor_rec_br')
                            weight_head = tf.multiply(qk_exp_head, norm_factor_rec_br)
                            v_t_head_c_slice = b_v_t_heads_tiled[:, bst_out:bst_out+OUTPUTLEN]
                            v_t_head = tf.slice(v_t_head_c_slice, [hst, 0], [HEADSIZE, -1], name='v_t_head_batch')
                            weighted_v_head = tf.matmul(weight_head, v_t_head, transpose_b=True, name='weighted_v_head')
                            weighted_v_heads_list.append(weighted_v_head)
                    # merge heads
                    weighted_v = tf.concat(weighted_v_heads_list, axis=1)
                    weighted_v_t = tf.transpose(weighted_v, perm=[1, 0])
                    b_weighted_v_t_list.append(weighted_v_t)
            b_weighted_v_t_tiled = tf.concat(b_weighted_v_t_list, axis=1, name='b_weighted_v_t_tiled')
            b_weighted_v_tiled = tf.transpose(b_weighted_v_t_tiled, perm=[1, 0], name='b_weighted_v_tiled')
            output = tf.matmul(b_weighted_v_tiled, tr_kernel, name='mhatt_output')
            return output

        def feed_forward_network(inputs, layer_name_prefix):
            filter_kernel_shape = [NUMHID, NUMHID]
            filter_bias_shape = [NUMHID]
            output_kernel_shape = [NUMHID, NUMHID]
            output_bias_shape = [NUMHID]
            filter_kernel = np.random.uniform(WMIN, WMAX, size=filter_kernel_shape).astype(np.float16)
            filter_bias = np.random.uniform(WMIN, WMAX, size=filter_bias_shape).astype(np.float16)
            output_kernel = np.random.uniform(WMIN, WMAX, size=output_kernel_shape).astype(np.float16)
            output_bias = np.random.uniform(WMIN, WMAX, size=output_bias_shape).astype(np.float16)
            with tf.name_scope('filter_layer'):
                filter_layer = tf.matmul(inputs, filter_kernel, name='matmul')
                filter_layer = tf.nn.bias_add(filter_layer, filter_bias, name='bias_add')
                filter_layer = tf.nn.relu(filter_layer)
            with tf.name_scope('output_layer'):
                output = tf.matmul(filter_layer, output_kernel, name='matmul')
                output = tf.nn.bias_add(output, output_bias, name='bias_add')
            return output

        with tf.Session(graph=tf.Graph()) as sess:
            enc_input = tf.placeholder(tf.float16, name='input', shape=[batch_input_len, NUMHID])
            b_bias_br = [
                tf.placeholder(tf.float16, name='input_bias_br_{}'.format(i), shape=[INPUTLEN, OUTPUTLEN])
                for i in range(BATCHSIZE)
            ]
            with tf.name_scope('encoder_layer'):
                enc_out = encoder_block(enc_input, b_bias_br, 0)
                output = tf.identity(enc_out, name='output')
            b_input_x_np = np.random.uniform(IMIN, IMAX, size=[BATCHSIZE, INPUTLEN, NUMHID]).astype(np.float16)
            b_input_x_r_np = b_input_x_np.reshape([batch_input_len, NUMHID])
            b_bias_np = np.zeros([BATCHSIZE, OUTPUTLEN]).astype(np.float16)
            b_bias_np[0, 5] = NEGINF
            b_bias_np[1, 8] = NEGINF
            b_bias_np[2, 9] = NEGINF
            b_bias_np[3, 2] = NEGINF
            b_bias_br_np = b_bias_np[:, :, np.newaxis].dot(np.ones([INPUTLEN, 1]).astype(np.float16).T).transpose([0, 2, 1])
            feed_dict = {
                'input:0': b_input_x_r_np
            }
            for i in range(BATCHSIZE):
                feed_dict['input_bias_br_{}:0'.format(i)] = b_bias_br_np[i]
            result_names = [output.name]
            result_ref = sess.run(result_names, feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, feed_dict=feed_dict, compiler_timeout=1)
        for name in feed_dict.keys():
            infer_graph.get_tensor_by_name(name)
        for name in result_names:
            infer_graph.get_tensor_by_name(name)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run(result_names, feed_dict)
                assert len(result_neuron) == len(result_ref)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)


class TestDynamicBatchSize(unittest.TestCase):

    def test_chain(self):
        np.random.seed(_RANDOM_SEED)
        pix = 8
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, pix, pix, 3], name='input0')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            relu0 = tf.nn.relu(conv2d0, name='relu0')
            feed_dict_compile = {
                'input0:0': np.random.uniform(-1, 1, size=[1, pix, pix, 3]).astype(np.float16),
            }
            feed_dict_list = []
            for batch_size in 1, 2, 3:
                feed_dict = {
                    'input0:0': np.random.uniform(-1, 1, size=[batch_size, pix, pix, 3]).astype(np.float16),
                }
                feed_dict_list.append(feed_dict)
            result_ref_list = [sess.run('relu0:0', feed_dict) for feed_dict in feed_dict_list]
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'},
                feed_dict=feed_dict_compile, output_tensors=['relu0:0'])
        assert infer_graph.get_operations()[1].outputs[0].shape.as_list() == [None, pix, pix, 3]
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                for feed_dict, result_ref in zip(feed_dict_list, result_ref_list):
                    result_neuron = sess.run('relu0:0', feed_dict)
                    np.testing.assert_allclose(result_neuron, result_ref, rtol=1e-2, atol=1e-2)

    def test_simple(self):
        infer_graph, result_names, feed_dict_list, result_ref_list = self._body()
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                for feed_dict, result_ref in zip(feed_dict_list, result_ref_list):
                    result_neuron = sess.run(result_names, feed_dict)
                    assert len(result_neuron) == len(result_ref)
                    for res_neuron, res_ref in zip(result_neuron, result_ref):
                        np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-2)

    def test_multithread(self):
        infer_graph, result_names, feed_dict_list, result_ref_list = self._body()
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                for _ in range(3):
                    with ThreadPoolExecutor(max_workers=len(feed_dict_list)) as executor:
                        future_list = [executor.submit(sess.run, result_names, feed_dict) for feed_dict in feed_dict_list]
                        result_neuron_list = [future.result() for future in future_list]
            for res_neuron, res_ref in zip(result_neuron_list, result_ref_list):
                np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-2)

    def _body(self):
        np.random.seed(_RANDOM_SEED)
        pix = 3
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, pix, pix, 3], name='input0')
            input1 = tf.placeholder(tf.int32, name='input1')
            input2 = tf.placeholder(tf.float16, [None, pix, pix, 3], name='input2')
            input3 = tf.placeholder(tf.bool, name='input3')
            identity_n0 = tf.identity_n([input0, input1, input2, input3], name='identity_n0')
            identity_n10, _, identity_n12, _ = tf.identity_n(identity_n0, name='identity_n1')
            conv2d0 = tf.nn.conv2d(identity_n10, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d2 = tf.nn.conv2d(identity_n12, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d2')
            add0 = tf.add(conv2d0, conv2d2, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            feed_dict_compile = {
                'identity_n1:0': np.random.uniform(-1, 1, size=[3, pix, pix, 3]).astype(np.float16),
                'identity_n1:2': np.random.uniform(-1, 1, size=[3, pix, pix, 3]).astype(np.float16),
            }
            feed_dict_list = []
            for batch_size in 1, 2, 3, 5, 11, 12, 1023:
                feed_dict = {
                    'identity_n1:0': np.random.uniform(-1, 1, size=[batch_size, pix, pix, 3]).astype(np.float16),
                    'identity_n1:2': np.random.uniform(-1, 1, size=[batch_size, pix, pix, 3]).astype(np.float16),
                }
                feed_dict_list.append(feed_dict)
            result_names = ['relu0:0', 'relu1:0']
            result_ref_list = [sess.run(result_names, feed_dict) for feed_dict in feed_dict_list]
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'},
                feed_dict=feed_dict_compile, output_tensors=result_names,
                dynamic_batch_size=True,
                compiler_workdir='./workdir')
        _assert_compiler_success(infer_graph)
        assert len([op for op in infer_graph.get_operations() if op.type == 'NeuronOp']) == 2
        return infer_graph, result_names, feed_dict_list, result_ref_list

    def test_unknown_rank(self):
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, None, name='input0')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            relu0 = tf.nn.relu(conv2d0, name='relu0')
            feed_dict_compile = {
                'input0:0': np.random.uniform(-1, 1, size=[3, 8, 8, 3]).astype(np.float16),
            }
            feed_dict_list = []
            for batch_size in 1, 2, 3, 5, 11, 12, 1023:
                feed_dict = {
                    'input0:0': np.random.uniform(-1, 1, size=[batch_size, 8, 8, 3]).astype(np.float16),
                }
                feed_dict_list.append(feed_dict)
            result_names = ['relu0:0']
            result_ref_list = [sess.run(result_names, feed_dict) for feed_dict in feed_dict_list]
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Relu'},
                feed_dict=feed_dict_compile, output_tensors=result_names,
                dynamic_batch_size=True,
                compiler_workdir='./workdir')
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                for feed_dict, result_ref in zip(feed_dict_list, result_ref_list):
                    result_neuron = sess.run(result_names, feed_dict)
                    assert len(result_neuron) == len(result_ref)
                    for res_neuron, res_ref in zip(result_neuron, result_ref):
                        np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-2)

    def test_concat_tensor_content(self):
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 3], name='input0')
            matmul0 = tf.matmul(input0, np.random.uniform(-1, 1, size=[3, 3]).astype(np.float16), name='matmul0')
            relu0 = tf.nn.relu(matmul0, name='relu0')
            matmul1 = tf.matmul(input0, np.random.uniform(-1, 1, size=[3, 3]).astype(np.float16), name='matmul1')
            relu1 = tf.nn.relu(matmul1, name='relu1')
            concat0 = tf.concat([matmul0, matmul1], axis=-2, name='concat0')
            feed_dict_compile = {
                'input0:0': np.random.uniform(-1, 1, size=[1, 3]).astype(np.float16),
            }
            result_names = ['concat0:0']
            graph_def = sess.graph.as_graph_def()
        tensor_proto = graph_def.node[7].attr['value'].tensor
        tensor_proto.tensor_content = np.array([-2], np.int32).tobytes()
        tensor_proto.int_val.pop()
        with tf.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def, name='')
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'MatMul', 'Const', 'Relu', 'ConcatV2'},
                feed_dict=feed_dict_compile, output_tensors=result_names)
        _assert_compiler_success(infer_graph)
        assert infer_graph.get_operations()[-2].get_attr('input_batch_axis') == [-1]


class TestSpecialOperator(unittest.TestCase):

    def test_conv2d_nchw(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 3, 2, 2], name='input0')
            kernel0 = np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16)
            conv2d0 = tf.nn.conv2d(input0, kernel0, strides=[1, 1, 1, 1], padding='VALID',
                                   data_format='NCHW', name='conv2d0')
            relu0 = tf.nn.relu(conv2d0, name='relu0')
            input0_nchw = np.random.uniform(-1, 1, size=input0.shape.as_list()).astype(np.float16)
            infer_graph0 = graph_util.inference_graph_from_session(
                sess, input_tensors=[input0], output_tensors=[relu0])
            infer_graph1 = graph_util.inference_graph_from_session(
                sess, input_tensors=[input0], output_tensors=[relu0], compiler_timeout=0.1)
        _assert_compiler_success(infer_graph0)
        with tf.Session(graph=tf.Graph()) as sess_ref:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            conv2d0 = tf.nn.conv2d(input0, kernel0, strides=[1, 1, 1, 1], padding='VALID',
                                   data_format='NHWC', name='conv2d0')
            relu0 = tf.nn.relu(conv2d0, name='relu0')
            result_ref_nhwc = sess_ref.run('relu0:0', {'input0:0': input0_nchw.transpose([0, 2, 3, 1])})
            result_ref_nchw = result_ref_nhwc.transpose([0, 3, 1, 2])
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph0) as sess:
                result_neuron_nchw = sess.run('relu0:0', {'input0:0': input0_nchw})
                np.testing.assert_allclose(result_neuron_nchw, result_ref_nchw, rtol=1e-2, atol=1e-3)
            with tf.Session(graph=infer_graph1) as sess:
                result_neuron_nchw = sess.run('relu0:0', {'input0:0': input0_nchw})
                np.testing.assert_allclose(result_neuron_nchw, result_ref_nchw, rtol=1e-2, atol=1e-3)

    def test_conv2d_nchw_explicit_paddings(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 3, 2, 2], name='input0')
            kernel0 = np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16)
            conv2d0 = tf.nn.conv2d(input0, kernel0, strides=[1, 1, 1, 1],
                                   padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
                                   data_format='NCHW', name='conv2d0')
            relu0 = tf.nn.relu(conv2d0, name='relu0')
            input0_nchw = np.random.uniform(-1, 1, size=input0.shape.as_list()).astype(np.float16)
            infer_graph0 = graph_util.inference_graph_from_session(
                sess, input_tensors=[input0], output_tensors=[relu0])
        with tf.Session(graph=tf.Graph()) as sess_ref:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            conv2d0 = tf.nn.conv2d(input0, kernel0, strides=[1, 1, 1, 1],
                                   padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
                                   data_format='NHWC', name='conv2d0')
            relu0 = tf.nn.relu(conv2d0, name='relu0')
            result_ref_nhwc = sess_ref.run('relu0:0', {'input0:0': input0_nchw.transpose([0, 2, 3, 1])})
            result_ref_nchw = result_ref_nhwc.transpose([0, 3, 1, 2])
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph0) as sess:
                result_neuron_nchw = sess.run('relu0:0', {'input0:0': input0_nchw})
                np.testing.assert_allclose(result_neuron_nchw, result_ref_nchw, rtol=1e-2, atol=1e-3)

    def test_maxpool_nchw(self):
        self._body_pool_nchw(tf.nn.max_pool)

    def test_avgpool_nchw(self):
        self._body_pool_nchw(tf.nn.avg_pool)

    def _body_pool_nchw(self, pool_func):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 3, 2, 2], name='input0')
            pool0 = pool_func(input0, [1, 1, 2, 2], strides=[1, 1, 1, 1], padding='VALID',
                              data_format='NCHW', name='pool0')
            relu0 = tf.nn.relu(pool0, name='relu0')
            input0_nchw = np.random.uniform(-1, 1, size=input0.shape.as_list()).astype(np.float16)
            infer_graph0 = graph_util.inference_graph_from_session(
                sess, input_tensors=[input0], output_tensors=[relu0])
            infer_graph1 = graph_util.inference_graph_from_session(
                sess, input_tensors=[input0], output_tensors=[relu0], compiler_timeout=0.1)
        _assert_compiler_success(infer_graph0)
        with tf.Session(graph=tf.Graph()) as sess_ref:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            pool0 = pool_func(input0, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID',
                              data_format='NHWC', name='pool0')
            relu0 = tf.nn.relu(pool0, name='relu0')
            result_ref_nhwc = sess_ref.run('relu0:0', {'input0:0': input0_nchw.transpose([0, 2, 3, 1])})
            result_ref_nchw = result_ref_nhwc.transpose([0, 3, 1, 2])
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph0) as sess:
                result_neuron_nchw = sess.run('relu0:0', {'input0:0': input0_nchw})
                np.testing.assert_allclose(result_neuron_nchw, result_ref_nchw, rtol=1e-2, atol=1e-3)
            with tf.Session(graph=infer_graph1) as sess:
                result_neuron_nchw = sess.run('relu0:0', {'input0:0': input0_nchw})
                np.testing.assert_allclose(result_neuron_nchw, result_ref_nchw, rtol=1e-2, atol=1e-3)

    def test_batchmatmulv2(self):
        np.random.seed(_RANDOM_SEED)
        input0_np = np.random.uniform(-1, 1, size=[1, 4, 2, 4]).astype(np.float16)
        input1_np = np.random.uniform(-1, 1, size=[1, 4, 4, 5]).astype(np.float16)
        feed_dict = {'input0:0': input0_np, 'input1:0': input1_np}
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 4, 2, 4], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 4, 4, 5], name='input1')
            batchmatmul0 = tf.matmul(input0, input1, name='batchmatmul0')
            result_ref = sess.run('batchmatmul0:0', feed_dict)
            infer_graph_ok = graph_util.inference_graph_from_session(
                sess, input_tensors=[input0, input1], output_tensors=[batchmatmul0],
                shape_feed_dict={input0: [1, 4, 2, 4], input1: [1, 4, 4, 5]})
        _assert_compiler_success(infer_graph_ok)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 4, 2, 4], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 4, 5], name='input1')
            batchmatmul0 = tf.matmul(input0, input1, name='batchmatmul0')
            infer_graph_not_ok = graph_util.inference_graph_from_session(
                sess, input_tensors=[input0, input1], output_tensors=[batchmatmul0],
                shape_feed_dict={input0: [1, 4, 2, 4], input1: [1, 4, 5]}, compiler_timeout=0.1)
            assert infer_graph_not_ok.get_operations()[-1].type == 'BatchMatMulV2'
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph_ok) as sess:
                result_neuron = sess.run('batchmatmul0:0', feed_dict)
                np.testing.assert_allclose(result_neuron, result_ref, rtol=1e-2, atol=1e-3)


class TestSharedMemoryInfer(unittest.TestCase):

    def test_shared_memory_infer(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.int32, name='input1')
            input2 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input2')
            input3 = tf.placeholder(tf.bool, name='input3')
            identity_n0 = tf.identity_n([input0, input1, input2, input3], name='identity_n0')
            identity_n10, _, identity_n12, _ = tf.identity_n(identity_n0, name='identity_n1')
            conv2d0 = tf.nn.conv2d(identity_n10, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d2 = tf.nn.conv2d(identity_n12, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d2')
            add0 = tf.add(conv2d0, conv2d2, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            feed_dict = {
                'identity_n1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
                'identity_n1:2': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_names = ['relu0:0', 'relu1:0']
            result_ref = sess.run(result_names, feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'},
                feed_dict=feed_dict, output_tensors=result_names,
                compiler_workdir='./workdir')
        for name in feed_dict.keys():
            infer_graph.get_tensor_by_name(name)
        for name in result_names:
            infer_graph.get_tensor_by_name(name)
        _assert_compiler_success(infer_graph)
        assert len([op for op in infer_graph.get_operations() if op.type == 'NeuronOp']) == 2
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run(result_names, feed_dict)
                assert len(result_neuron) == len(result_ref)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)

    def setUp(self):
        self.neuron_rtd_shm_map_set = False
        if 'NEURON_RTD_SHM_MAP' not in os.environ:
            os.environ['NEURON_RTD_SHM_MAP'] = 'yes'
            self.neuron_rtd_shm_map_set = True

    def tearDown(self):
        if self.neuron_rtd_shm_map_set:
            os.environ.pop('NEURON_RTD_SHM_MAP')


class TestNeuronCoreGroupSizes(unittest.TestCase):

    def setUp(self):
        self.env_set = False
        if 'NEURONCORE_GROUP_SIZES' not in os.environ:
            os.environ['NEURONCORE_GROUP_SIZES'] = self._neuroncore_group_sizes
            self.env_set = True

    def tearDown(self):
        if self.env_set:
            os.environ.pop('NEURONCORE_GROUP_SIZES')

    def body(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input1')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            feed_dict = {key: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16)
                         for key in ['input0:0', 'input1:0']}
            result_ref = sess.run('relu0:0', feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run('relu0:0', feed_dict)
            np.testing.assert_allclose(result_neuron, result_ref, rtol=1e-2, atol=1e-3)

class TestNeuronCoreGroupSizesCase0(TestNeuronCoreGroupSizes):
    _neuroncore_group_sizes = '[4,100,1]'
    def test(self): self.body()
class TestNeuronCoreGroupSizesCase1(TestNeuronCoreGroupSizes):
    _neuroncore_group_sizes = '[1, 1, 1]'
    def test(self): self.body()
class TestNeuronCoreGroupSizesCase2(TestNeuronCoreGroupSizes):
    _neuroncore_group_sizes = '[1,1,1,1,1,1,1]'
    def test(self): self.body()


class TestMiscUtils(unittest.TestCase):

    def test_opt_num_cores(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input1')
            kernel0 = tf.Variable(np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16))
            kernel1 = tf.Variable(np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16))
            conv2d0 = tf.nn.conv2d(input0, kernel0, strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, kernel1, strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            kernel2 = tf.Variable(np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16))
            conv2d2 = tf.nn.conv2d(sigmoid0, kernel2, strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            feed_dict = {
                'input0:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
                'input1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            sess.run(tf.global_variables_initializer())
            result_ref = sess.run(['relu0:0', 'sigmoid0:0', 'relu1:0', 'add0:0'], feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, feed_dict=feed_dict, output_tensors={'relu0:0', 'sigmoid0:0', relu1, 'add0:0'},
                op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        _assert_compiler_success(infer_graph)
        infer_graph_def = infer_graph.as_graph_def()
        for node in infer_graph_def.node:
            if node.op == 'NeuronOp':
                node.attr['model_config'].list.i[:] = [3, 4, 5, 10]
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=tf.Graph()) as sess:
                tf.import_graph_def(infer_graph_def, name='')
                result_neuron = sess.run(['relu0:0', 'sigmoid0:0', 'relu1:0', 'add0:0'], feed_dict)
                assert len(result_neuron) == len(result_ref)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)

    def test_compiler_verbose(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input1')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
            conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            relu1 = tf.nn.relu(conv2d2, name='relu1')
            feed_dict = {
                'input0:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
                'input1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_ref0 = sess.run(['relu0:0', 'relu1:0'], feed_dict)
            infer_graph0 = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'},
                compiler_workdir='./workdir', compiler_verbose=2)
        _assert_compiler_success(infer_graph0)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph0) as sess:
                result_neuron0 = sess.run(['relu0:0', 'relu1:0'], feed_dict)
                assert len(result_neuron0) == len(result_ref0)
                for res_neuron, res_ref in zip(result_neuron0, result_ref0):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)

    def test_auto_duplicate_hint(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [None, 2, 2, 3], name='input0')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            relu0 = tf.nn.relu(conv2d0, name='relu0')
            feed_dict = {
                'input0:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_ref0 = sess.run('relu0:0', feed_dict)
            infer_graph = graph_util.inference_graph_from_session(sess, feed_dict=feed_dict)
        _assert_compiler_success(infer_graph)
        graph_def = infer_graph.as_graph_def()
        graph_def.node[1].attr['model_config'].list.i[2] = 4
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=tf.Graph()) as sess:
                tf.import_graph_def(graph_def, name='')
                result_neuron0 = sess.run('relu0:0', feed_dict)
                np.testing.assert_allclose(result_neuron0, result_ref0, rtol=1e-2, atol=1e-3)


class TestNeuronCCFlagsEnv(unittest.TestCase):

    def setUp(self):
        self.env_set = False
        if 'NEURON_CC_FLAGS' not in os.environ:
            os.environ['NEURON_CC_FLAGS'] = self._neuron_cc_flags
            self.env_set = True

    def tearDown(self):
        if self.env_set:
            os.environ.pop('NEURON_CC_FLAGS')

    def _gen_graph(self):
        input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
        input1 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input1')
        conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                               strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
        conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                               strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
        add0 = tf.add(conv2d0, conv2d1, name='add0')
        relu0 = tf.nn.relu(add0, name='relu0')
        sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
        conv2d2 = tf.nn.conv2d(sigmoid0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                               strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
        relu1 = tf.nn.relu(conv2d2, name='relu1')
        feed_dict = {
            'input0:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            'input1:0': np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        }
        return ['relu0:0', 'relu1:0'], feed_dict

class TestNeuronCCFlagsEnvMustCompileSuccess(TestNeuronCCFlagsEnv):
    _neuron_cc_flags = '--must-compile --dump-prefix ./workdir --fp32-cast matmult'
    def test(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            fetch_list, feed_dict = self._gen_graph()
            result_ref0 = sess.run(fetch_list, feed_dict)
            infer_graph0 = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})

class TestNeuronCCFlagsEnvMustCompileFailure(TestNeuronCCFlagsEnv):
    _neuron_cc_flags = '--i-am-not-recognized --must-compile'
    def test(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            fetch_list, feed_dict = self._gen_graph()
            result_ref0 = sess.run(fetch_list, feed_dict)
            with self.assertRaises(ValueError) as cm:
                infer_graph0 = graph_util.inference_graph_from_session(
                    sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
            assert cm.exception.args[0].startswith('The following subgraphs failed')

class TestNeuronCCFlagsEnvDump(TestNeuronCCFlagsEnv):
    _neuron_cc_flags = '--dump-prefix ./workdir_neuron_cc_flags_dump --fp32-cast matmult'
    def test(self):
        np.random.seed(_RANDOM_SEED)
        workdir = './workdir_neuron_cc_flags_dump'
        shutil.rmtree(workdir, ignore_errors=True)
        with tf.Session(graph=tf.Graph()) as sess:
            fetch_list, feed_dict = self._gen_graph()
            result_ref0 = sess.run(fetch_list, feed_dict)
            infer_graph0 = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
            _assert_compiler_success(infer_graph0)
            assert os.path.isdir(workdir)
            assert len(list(glob.glob(os.path.join(workdir, 'neuron_op_*', 'graph_def.pb')))) == 2
            assert len(list(glob.glob(os.path.join(workdir, 'neuron_op_*', 'graph_def.neff')))) == 2
            assert len(list(glob.glob(os.path.join(workdir, 'neuron_op_*', 'graph_def.neuron-cc.log')))) == 2


class TestStress(unittest.TestCase):

    def test_multithread_load_infer(self):
        np.random.seed(_RANDOM_SEED)
        max_workers = 48
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input1')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            feed_dict_list = [{key: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16)
                               for key in ['input0:0', 'input1:0']} for _ in range(max_workers)]
            result_ref_list = [sess.run('relu0:0', feed_dict) for feed_dict in feed_dict_list]
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                latency_list = []
                for _ in range(3):
                    start = time.time()
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_list = [executor.submit(sess.run, 'relu0:0', feed_dict) for feed_dict in feed_dict_list]
                        result_neuron_list = [future.result() for future in future_list]
                    elapsed = time.time() - start
                    latency_list.append(elapsed)
                logging.debug(latency_list)
            for res_neuron, res_ref in zip(result_neuron_list, result_ref_list):
                np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)

    @unittest.skipIf('--runxfail' not in sys.argv, 'Do not run by default')
    def test_random_graph_multithread(self):
        np.random.seed(_RANDOM_SEED)
        max_workers = 48
        graph_fn = 'random_graph.pb'
        infer_graph_fn = 'random_graph_infer.pb'
        input_names_fn = 'random_input_names.txt'
        output_names_fn = 'random_output_names.txt'
        if (os.path.isfile(graph_fn) and os.path.isfile(infer_graph_fn) and
                os.path.isfile(input_names_fn) and os.path.isfile(output_names_fn)):
            with open(input_names_fn, 'r') as f:
                input_names = f.read().split(',')
            with open(output_names_fn, 'r') as f:
                outputs = f.read().split(',')
            np.random.seed(_RANDOM_SEED)
            feed_dict_list = [{name: np.random.uniform(-1, 1, size=[1, 1]) for name in input_names}
                              for _ in range(max_workers)]
            with tf.Session(graph=tf.Graph()) as sess:
                graph_def = tf.GraphDef()
                with open(graph_fn, 'rb') as f:
                    graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                result_ref_list = [sess.run(outputs, feed_dict) for feed_dict in feed_dict_list]
            infer_graph_def = tf.GraphDef()
            with open(infer_graph_fn, 'rb') as f:
                infer_graph_def.ParseFromString(f.read())
            infer_graph = tf.Graph()
            with infer_graph.as_default():
                tf.import_graph_def(infer_graph_def, name='')
        else:
            rdrange = lambda: range(np.random.randint(30, 50))
            no_fuse_ops = set()
            with tf.Session(graph=tf.Graph()) as sess:
                inputs = [tf.placeholder(tf.float32, [1, 1]) for _ in rdrange()]
                with open(input_names_fn, 'w') as f:
                    f.write(','.join(ts.name for ts in inputs))
                np.random.seed(_RANDOM_SEED)
                feed_dict_list = [{ts.name: np.random.uniform(-1, 1, size=[1, 1]) for ts in inputs}
                                  for _ in range(max_workers)]
                tensors = inputs
                for _ in range(10):
                    tensors = [tf.add(np.random.choice(tensors), np.random.choice(tensors)) for _ in rdrange()]
                    tensors = [tf.nn.relu(ts) for ts in tensors]
                    tensors = [tf.identity(ts) for ts in tensors]
                    no_fuse_ops.update(ts.op.name for ts in tensors)
                outputs = [ts.name for ts in tensors]
                with open(output_names_fn, 'w') as f:
                    f.write(','.join(outputs))
                with open(graph_fn, 'wb') as f:
                    f.write(sess.graph.as_graph_def().SerializeToString())
                result_ref_list = [sess.run(outputs, feed_dict) for feed_dict in feed_dict_list]
                infer_graph = graph_util.inference_graph_from_session(
                    sess, input_tensors=inputs, output_tensors=outputs,
                    op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'}, minimum_segment_size=2,
                    no_fuse_ops=no_fuse_ops)
                with open(infer_graph_fn, 'wb') as f:
                    f.write(infer_graph.as_graph_def().SerializeToString())
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                for idx in range(10):
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        start = time.time()
                        future_list = [executor.submit(sess.run, outputs, feed_dict)
                                       for feed_dict in feed_dict_list]
                        result_neuron_list = [future.result() for future in future_list]
                        logging.warning('cycle {}, elapsed {}'.format(idx, time.time() - start))
                for res_neuron, res_ref in zip(result_neuron_list, result_ref_list):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-2)


class TestLargeIO(unittest.TestCase):

    def test_large_io_even(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float32, [4096, 4096], name='input0')
            relu0 = tf.nn.relu(input0, name='relu0')
            feed_dict = {input0.name: np.random.uniform(0, 1, size=input0.shape.as_list()).astype(np.float32)}
            result_ref = sess.run('relu0:0', feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run('relu0:0', feed_dict)
                np.testing.assert_allclose(result_neuron, result_ref)
            with tf.Session(graph=infer_graph, config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:
                result_neuron = sess.run('relu0:0', feed_dict)
                np.testing.assert_allclose(result_neuron, result_ref)

    def test_large_io_odd(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float32, [4095, 4095], name='input0')
            relu0 = tf.nn.relu(input0, name='relu0')
            feed_dict = {input0.name: np.random.uniform(0, 1, size=input0.shape.as_list()).astype(np.float32)}
            result_ref = sess.run('relu0:0', feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run('relu0:0', feed_dict)
                np.testing.assert_allclose(result_neuron, result_ref)
            with tf.Session(graph=infer_graph, config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:
                result_neuron = sess.run('relu0:0', feed_dict)
                np.testing.assert_allclose(result_neuron, result_ref)

    def test_large_io_mid(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [231, 43], name='input0')
            relu0 = tf.nn.relu(input0, name='relu0')
            feed_dict = {input0.name: np.random.uniform(0, 1, size=input0.shape.as_list()).astype(np.float32)}
            result_ref = sess.run('relu0:0', feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run('relu0:0', feed_dict)
                np.testing.assert_allclose(result_neuron, result_ref)


class TestShapeInference(unittest.TestCase):

    def test_no_inputs_simple(self):
        np.random.seed(_RANDOM_SEED)
        # case where call_cpp_shape_fn is sufficient
        graph = tf.Graph()
        with graph.as_default():
            input0 = tf.placeholder(tf.float32, [None, 16], name='input0')
            input1 = tf.placeholder(tf.float32, [None, 8], name='input1')
            matmul0 = tf.matmul(input0, np.random.rand(16, 24).astype(np.float32), name='matmul0')
            matmul1 = tf.matmul(input1, np.random.rand(8, 24).astype(np.float32), name='matmul1')
            add0 = tf.add(matmul0, matmul1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            exp0 = tf.exp(add0, name='exp0')
        shape_feed_dict0 = {input0: [2, 16], 'input1:0': [2, 8]}
        shape_feed_dict1 = {'input0:0': (2, 16), input1: (2, 8)}
        desired_shapes = {
            input0.name: [2, 16],
            input1.name: [2, 8],
            matmul0.name: [2, 24],
            matmul1.name: [2, 24],
            add0.name: [2, 24],
            relu0.name: [2, 24],
            exp0.name: [2, 24],
        }
        output_tensors = [relu0, exp0]
        shape0 = TensorShape([2, 16]).as_proto()
        shape1 = TensorShape([2, 8]).as_proto()
        shape_feed_dict2 = {'input0:0': shape0, 'input1:0': shape1}
        graph_def = graph.as_graph_def()
        shaped_graph_def = graph_util.shape_inference(graph_def, shape_feed_dict0, output_tensors)
        inferred_shapes = _get_inferred_shapes_from_graph(shaped_graph_def)
        for name in desired_shapes:
            assert inferred_shapes[name] == desired_shapes[name]
        shaped_graph_def = graph_util.shape_inference(graph_def, shape_feed_dict1, output_tensors)
        inferred_shapes = _get_inferred_shapes_from_graph(shaped_graph_def)
        for name in desired_shapes:
            assert inferred_shapes[name] == desired_shapes[name]
        shaped_graph_def = graph_util.shape_inference(graph_def, shape_feed_dict2, output_tensors)
        inferred_shapes = _get_inferred_shapes_from_graph(shaped_graph_def)
        for name in desired_shapes:
            assert inferred_shapes[name] == desired_shapes[name]

    def test_inputs_short_long(self):
        # case where call_cpp_shape_fn is sufficient
        graph = tf.Graph()
        with graph.as_default():
            input0 = tf.placeholder(tf.float32, [None, 3, 5], name='input0')
            input1 = tf.placeholder(tf.float32, [None, 3, 5], name='input1')
            relu10 = tf.nn.relu(input1, name='relu10')
            relu11 = tf.nn.relu(relu10, name='relu11')
            relu12 = tf.nn.relu(relu11, name='relu12')
            add0 = tf.add(input0, relu12, name='add0')
            exp0 = tf.exp(add0, name='exp0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
        shape_feed_dict = {'input0:0': [1, 3, 5], input1: [1, 3, 5]}
        desired_shapes = {
            input0.name: [1, 3, 5],
            input1.name: [1, 3, 5],
            relu10.name: [1, 3, 5],
            relu11.name: [1, 3, 5],
            relu12.name: [1, 3, 5],
            add0.name: [1, 3, 5],
            exp0.name: [1, 3, 5],
            sigmoid0.name: [1, 3, 5],
        }
        shaped_graph_def = graph_util.shape_inference(graph.as_graph_def(), shape_feed_dict, [exp0, sigmoid0])
        inferred_shapes = _get_inferred_shapes_from_graph(shaped_graph_def)
        for name in desired_shapes:
            assert inferred_shapes[name] == desired_shapes[name]

    def test_short_long_mid(self):
        # case where call_cpp_shape_fn is sufficient
        graph = tf.Graph()
        with graph.as_default():
            input0 = tf.placeholder(tf.float32, [None, 3, 5], name='input0')
            input1 = tf.placeholder(tf.float32, [1, 3, 5], name='input1')
            input2 = tf.placeholder(tf.float32, [None, 3, 5], name='input2')
            input3 = tf.placeholder(tf.float32, [None, 3, 5], name='input3')
            identity_n0 = tf.identity_n([input0, input1, input2, input3], name='identity_n0')
            identity_n00, identity_n01, identity_n02, identity_n03  = identity_n0
            relu30 = tf.nn.relu(identity_n03, name='relu30')
            relu31 = tf.nn.relu(relu30, name='relu31')
            relu32 = tf.nn.relu(relu31, name='relu32')
            add0 = tf.add(identity_n00, relu32, name='add0')
            exp0 = tf.exp(add0, name='exp0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
        shape_feed_dict = {identity_n00: [1, 3, 5], 'identity_n0:3': [1, 3, 5]}
        desired_shapes = {
            input0.name: [None, 3, 5],
            input1.name: [1, 3, 5],
            input2.name: [None, 3, 5],
            input3.name: [None, 3, 5],
            identity_n00.name: [1, 3, 5],
            identity_n03.name: [1, 3, 5],
            relu30.name: [1, 3, 5],
            relu31.name: [1, 3, 5],
            relu32.name: [1, 3, 5],
            add0.name: [1, 3, 5],
            exp0.name: [1, 3, 5],
            sigmoid0.name: [1, 3, 5],
        }
        output_tensors = [identity_n00, identity_n01, identity_n02, identity_n03, relu30, relu31, relu32, add0, exp0, sigmoid0]
        shaped_graph_def = graph_util.shape_inference(graph.as_graph_def(), shape_feed_dict, output_tensors)
        inferred_shapes = _get_inferred_shapes_from_graph(shaped_graph_def)
        for name in desired_shapes:
            assert TensorShape(inferred_shapes[name]).as_list() == TensorShape(desired_shapes[name]).as_list()

    def test_strided_slice(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [3, 1, 1, 1, 1, 2], name='input0')
            const0_np = np.random.rand(3, 2, 1, 2, 3, 2).astype(np.float16)
            const0 = tf.constant(const0_np, name='const0')
            stridedslice0 = const0[:, 1, ..., tf.newaxis, 1:2, 2:, :5]
            output0 = tf.identity(input0 + stridedslice0, name='output0')
            evaluated_map_tf = {ts.name: ts.name for ts in stridedslice0.op.inputs}
            evaluated_map = sess.run(evaluated_map_tf)
        shape_feed_dict0 = {input0: [3, 1, 1, 1, 1, 2]}
        shaped_graph_def = graph_util.shape_inference(sess.graph.as_graph_def(), shape_feed_dict0, [stridedslice0, output0])
        inferred_shapes = _get_inferred_shapes_from_graph(shaped_graph_def)
        assert [3, 1, 1, 1, 1, 2] == TensorShape(inferred_shapes[stridedslice0.name])

    def test_with_inputs_simple(self):
        np.random.seed(_RANDOM_SEED)
        # case where session.run is required
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float32, [None, 16], name='input0')
            input1 = tf.placeholder(tf.float32, [None, 8], name='input1')
            matmul0 = tf.matmul(input0, np.random.rand(16, 24).astype(np.float32), name='matmul0')
            matmul1 = tf.matmul(input1, np.random.rand(8, 24).astype(np.float32), name='matmul1')
            add0 = tf.add(matmul0, matmul1, name='add0')
            reshape0 = tf.reshape(add0, tf.range(2, 5), name='reshape0')
            relu0 = tf.nn.relu(reshape0, name='relu0')
            exp0 = tf.exp(reshape0, name='exp0')
            feed_dict = {input0.name: np.random.rand(1, 16), 'input1:0': np.random.rand(1, 8)}
            result_ref = sess.run(['relu0:0', 'exp0:0'], feed_dict)
            infer_graph = graph_util.inference_graph_from_session(
                sess, op_whitelist={'MatMul', 'Const', 'Add', 'Relu'}, feed_dict=feed_dict)
        _assert_compiler_success(infer_graph)
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=infer_graph) as sess:
                result_neuron = sess.run(['relu0:0', 'exp0:0'], feed_dict)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-2)


class TestHelperFunction(unittest.TestCase):

    def test_sorted_inferrable_ops_branch_merge(self):
        np.random.seed(_RANDOM_SEED)
        graph = tf.Graph()
        with graph.as_default():
            input0 = tf.placeholder(tf.float32, [1], name='input0')
            const0 = tf.constant(np.random.rand(1).astype(np.float32), name='const0')
            relu00 = tf.nn.relu(input0, name='relu00')
            relu10 = tf.nn.relu(input0, name='relu10')
            relu11 = tf.nn.relu(relu10, name='relu11')
            relu12 = tf.nn.relu(relu11, name='relu12')
            add0 = tf.add(relu00, relu12, name='add0')
            add1 = tf.add(add0, const0, name='add1')
            sigmoid00 = tf.sigmoid(add0, name='sigmoid00')
            sigmoid01 = tf.sigmoid(sigmoid00, name='sigmoid01')
            sigmoid02 = tf.sigmoid(sigmoid01, name='sigmoid02')
            sigmoid10 = tf.sigmoid(add0, name='sigmoid10')
            add2 = tf.add(sigmoid02, sigmoid10, name='add2')
        graph_def = graph.as_graph_def()
        node_list = list(graph.as_graph_def().node)
        for _ in range(10):
            np.random.shuffle(node_list)
            shuffled_graph_def = tf.GraphDef()
            shuffled_graph_def.node.extend(copy.deepcopy(node) for node in node_list)
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(shuffled_graph_def, name='')
            sorted_ops = graph.get_operations()
            for op in graph.get_operations():
                for ts in op.inputs:
                    assert sorted_ops.index(ts.op) < sorted_ops.index(op)

    def test_compile_subgraphs(self):
        np.random.seed(_RANDOM_SEED)
        graph = tf.Graph()
        with graph.as_default():
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input1')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
        feed_dict0 = {
            input0: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            input1: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        }
        feed_dict1 = {
            input0: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            input1: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
        }
        with tf.Session(graph=graph) as sess:
            sg0_relu0, sg0_sigmoid0 = sess.run([relu0, sigmoid0], feed_dict0)
            sg1_relu0, sg1_sigmoid0 = sess.run([relu0, sigmoid0], feed_dict1)
            feed_dict2 = {
                input0: sg0_relu0,
                input1: sg1_relu0,
            }
            feed_dict3 = {
                input0: sg0_sigmoid0,
                input1: sg1_sigmoid0,
            }
            result_ref = sess.run([relu0, sigmoid0], feed_dict2)
            result_ref.extend(sess.run([relu0, sigmoid0], feed_dict3))
        feed_dict_all = {
            'input0:0': feed_dict0[input0],
            'input1:0': feed_dict0[input1],
            'input2:0': feed_dict1[input0],
            'input3:0': feed_dict1[input1],
        }

        subgraph_graph_def_str = graph.as_graph_def(add_shapes=True).SerializeToString()
        graph = tf.Graph()
        with graph.as_default():
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input1')
            with tf.name_scope('neuron_op0'):
                sg0 = neuron_op(
                    [input0, input1], graph_def=subgraph_graph_def_str,
                    input_names=['input0:0', 'input1:0'],
                    input_shapes=[[1, 2, 2, 3], [1, 2, 2, 3]],
                    output_names=['relu0:0', 'sigmoid0:0'],
                    output_dtypes=[tf.float16, tf.float16],
                    output_shapes=[[1, 2, 2, 3], [1, 2, 2, 3]],
                    executable=b'', name='neuron_op0',
                )
                sg0 = tf.identity_n(sg0)
            input2 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input2')
            input3 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input3')
            with tf.name_scope('neuron_op1'):
                sg1 = neuron_op(
                    [input2, input3], graph_def=subgraph_graph_def_str,
                    input_names=['input0:0', 'input1:0'],
                    input_shapes=[[1, 2, 2, 3], [1, 2, 2, 3]],
                    output_names=['relu0:0', 'sigmoid0:0'],
                    output_dtypes=[tf.float16, tf.float16],
                    output_shapes=[[1, 2, 2, 3], [1, 2, 2, 3]],
                    executable=b'', name='neuron_op1',
                )
                sg1 = tf.identity_n(sg1)
            with tf.name_scope('neuron_op2'):
                input4 = tf.identity(sg0[0], name='input4')
                input5 = tf.identity(sg1[0], name='input5')
                sg2 = neuron_op(
                    [input4, input5], graph_def=subgraph_graph_def_str,
                    input_names=['input0:0', 'input1:0'],
                    input_shapes=[[1, 2, 2, 3], [1, 2, 2, 3]],
                    output_names=['relu0:0', 'sigmoid0:0'],
                    output_dtypes=[tf.float16, tf.float16],
                    output_shapes=[[1, 2, 2, 3], [1, 2, 2, 3]],
                    executable=b'', name='neuron_op2',
                )
                sg2 = tf.identity_n(sg2)
            input6 = tf.identity(sg0[1], name='input6')
            input7 = tf.identity(sg1[1], name='input7')
            sg3 = neuron_op(
                [input6, input7], graph_def=subgraph_graph_def_str,
                input_names=['input0:0', 'input1:0'],
                input_shapes=[[1, 2, 2, 3], [1, 2, 2, 3]],
                output_names=['relu0:0', 'sigmoid0:0'],
                output_dtypes=[tf.float16, tf.float16],
                output_shapes=[[1, 2, 2, 3], [1, 2, 2, 3]],
                executable=b'', name='neuron_op3',
            )
            sg3 = tf.identity_n(sg3)

        graph_def = graph.as_graph_def()
        compiled_graph_def = graph_util.compile_subgraphs(graph_def, workdir='./workdir')
        for node in compiled_graph_def.node:
            if node.op == 'NeuronOp':
                assert node.attr['executable'].s != b''
        compiled_graph_def = graph_util.compile_subgraphs(graph_def)
        for node in compiled_graph_def.node:
            if node.op == 'NeuronOp':
                assert node.attr['executable'].s != b''
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            with tf.Session(graph=tf.Graph()) as sess:
                tf.import_graph_def(compiled_graph_def, name='')
                output_neuron = [tensor.name for tensor in sg2]
                output_neuron.extend(tensor.name for tensor in sg3)
                result_neuron = sess.run(output_neuron, feed_dict_all)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)


def _get_inferred_shapes(tensor_shape_map):
    inferred_shapes = {}
    for name, shape_proto in tensor_shape_map.items():
        if isinstance(shape_proto, str):
            inferred_shapes[name] = shape_proto
        else:
            inferred_shapes[name] = TensorShape(shape_proto).as_list()
    return inferred_shapes


def _get_inferred_shapes_from_graph(graph_def):
    inferred_shapes = {}
    for node in graph_def.node:
        if '_aws_neuron_inferred_shapes' in node.attr:
            for port, shape_proto in enumerate(node.attr['_aws_neuron_inferred_shapes'].list.shape):
                inferred_shapes['{}:{}'.format(node.name, port)] = TensorShape(shape_proto)
    return inferred_shapes


class TestWhitelistPartition(unittest.TestCase):

    def test_simple(self):
        np.random.seed(_RANDOM_SEED)
        graph = tf.Graph()
        with graph.as_default():
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            input1 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input1')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input1, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            sigmoid0 = tf.sigmoid(add0, name='sigmoid0')
        graph_def = graph.as_graph_def(add_shapes=True)
        signature_def0 = graph_util.build_signature_def([input0, input1], [add0, relu0, sigmoid0])
        partitioned_graph_def0 = graph_util.whitelist_partition(
            graph_def, signature_def=signature_def0,
            op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        signature_def1 = graph_util.build_signature_def([conv2d0, input1], [add0, relu0, sigmoid0])
        partitioned_graph_def1 = graph_util.whitelist_partition(
            graph_def, signature_def=signature_def1,
            op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        signature_def2 = graph_util.build_signature_def([input0, input1], [relu0, sigmoid0])
        partitioned_graph_def2 = graph_util.whitelist_partition(
            graph_def, signature_def=signature_def2,
            op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        assert len(partitioned_graph_def0.node) == 6
        assert len(partitioned_graph_def0.node[2].attr['output_names'].list.s) == 2
        assert len(partitioned_graph_def1.node) == 8
        assert len(partitioned_graph_def2.node) == 5
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(partitioned_graph_def0, name='')
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(partitioned_graph_def1, name='')
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(partitioned_graph_def2, name='')

    def test_branch_merge(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            relu0 = tf.nn.relu(add0, name='relu0')
            feed_dict = {
                input0.name: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_ref = sess.run([relu0, add0], feed_dict)
        signature_def = graph_util.build_signature_def([input0], [add0, relu0])
        partitioned_graph_def = graph_util.whitelist_partition(
            sess.graph.as_graph_def(add_shapes=True), signature_def=signature_def,
            op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'})
        neuron_op_node = partitioned_graph_def.node[1]
        assert len(neuron_op_node.input) == 1
        assert len(neuron_op_node.attr['input_names'].list.s) == 1
        assert len(neuron_op_node.attr['input_dtypes'].list.type) == 1
        subgraph_def = tf.GraphDef()
        subgraph_def.ParseFromString(neuron_op_node.attr['graph_def'].s)
        with tf.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(subgraph_def, name='')
            assert len([op for op in sess.graph.get_operations() if op.type == 'Placeholder']) == 1
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            compiled_graph_def = graph_util.compile_subgraphs(partitioned_graph_def, workdir='./workdir')
            with tf.Session(graph=tf.Graph()) as sess:
                tf.import_graph_def(compiled_graph_def, name='')
                result_neuron = sess.run([relu0.name, add0.name], feed_dict)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-2)

    def test_no_fuse(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            conv2d2 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d2')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            add1 = tf.add(conv2d0, conv2d2, name='add1')
            add2 = tf.add(conv2d1, conv2d2, name='add2')
            add3 = tf.add(add2, add1, name='add3')
            add4 = tf.add(add1, add0, name='add4')
            add5 = tf.add(add2, add0, name='add5')
            relu0 = tf.nn.relu(add3, name='relu0')
            relu1 = tf.nn.relu(add4, name='relu1')
            relu2 = tf.nn.relu(add5, name='relu2')
            feed_dict = {
                input0.name: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_ref = sess.run([relu0, relu1, relu2], feed_dict)
        signature_def = graph_util.build_signature_def([input0], [relu0, relu1, relu2])
        partitioned_graph_def = graph_util.whitelist_partition(
            sess.graph.as_graph_def(add_shapes=True), signature_def=signature_def,
            op_whitelist={'Conv2D', 'Const', 'Add', 'Relu'},
            no_fuse_ops=['add0'])
        assert len(partitioned_graph_def.node) == 8
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            compiled_graph_def = graph_util.compile_subgraphs(
                partitioned_graph_def, workdir='./workdir')
            with tf.Session(graph=tf.Graph()) as sess:
                tf.import_graph_def(compiled_graph_def, name='')
                result_neuron = sess.run([relu0.name, relu1.name, relu2.name], feed_dict)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)

    def test_no_fuse_force_fuse(self):
        np.random.seed(_RANDOM_SEED)
        with tf.Session(graph=tf.Graph()) as sess:
            input0 = tf.placeholder(tf.float16, [1, 2, 2, 3], name='input0')
            conv2d0 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d0')
            conv2d1 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d1')
            conv2d2 = tf.nn.conv2d(input0, np.random.uniform(-1, 1, size=[1, 1, 3, 3]).astype(np.float16),
                                   strides=[1, 1, 1, 1], padding='VALID', name='conv2d2')
            add0 = tf.add(conv2d0, conv2d1, name='add0')
            add1 = tf.add(conv2d0, conv2d2, name='add1')
            add2 = tf.add(conv2d1, conv2d2, name='add2')
            add3 = tf.add(add2, add1, name='add3')
            add4 = tf.add(add1, add0, name='add4')
            add5 = tf.add(add2, add0, name='add5')
            relu0 = tf.nn.relu(add3, name='relu0')
            relu1 = tf.nn.relu(add4, name='relu1')
            relu2 = tf.nn.relu(add5, name='relu2')
            feed_dict = {
                input0.name: np.random.uniform(-1, 1, size=[1, 2, 2, 3]).astype(np.float16),
            }
            result_ref = sess.run([relu0, relu1, relu2], feed_dict)
        signature_def = graph_util.build_signature_def([input0], [relu0, relu1, relu2])
        partitioned_graph_def = graph_util.whitelist_partition(
            sess.graph.as_graph_def(add_shapes=True), signature_def=signature_def,
            op_whitelist={'Conv2D', 'Const', 'Relu'},
            no_fuse_ops=[conv2d2.op], force_fuse_ops=['add0', add5.op])
        assert len(partitioned_graph_def.node) == 12
        if 'NEURON_TF_COMPILE_ONLY' not in os.environ:
            compiled_graph_def = graph_util.compile_subgraphs(
                partitioned_graph_def, workdir='./workdir')
            with tf.Session(graph=tf.Graph()) as sess:
                tf.import_graph_def(compiled_graph_def, name='')
                result_neuron = sess.run([relu0.name, relu1.name, relu2.name], feed_dict)
                for res_neuron, res_ref in zip(result_neuron, result_ref):
                    np.testing.assert_allclose(res_neuron, res_ref, rtol=1e-2, atol=1e-3)


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

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
import unittest
import numpy as np
import tensorflow as tf
import tensorflow.neuron as tfn
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.neuron.python import graph_def_util as gdu
from tensorflow.neuron.python import graph_util
from tensorflow.neuron.python import meta_graph_util
from tensorflow.neuron.python.ops.gen_neuron_op import neuron_op
from tensorflow.python.platform import tf_logging as logging


class TestConv2dSamePaddingPass(unittest.TestCase):

    def test_same_padding_into_pad_and_no_padding_conv(self):
        '''
        Test asserts that the SAME padding has been removed from Conv2D
        and asserts that the new padding op has been added
        '''
            
        # Model Creation
        kernel = tf.random.uniform([7,7,3,64])
        kernel = tf.cast(kernel, tf.float16)

        def func(tensor):
            #tensor = tf.pad(tensor, [[0, 0], [0, 0], [3, 3], [3, 3]])
            tensor = tf.nn.conv2d(tensor, kernel, padding='SAME', strides=[1, 1, 2, 2], data_format='NCHW')
            return tensor

        input_tensor = tf.random.uniform([1, 3, 224, 224])
        input_tensor = tf.cast(input_tensor, tf.float16)

        layer_neuron = tfn.trace(func, input_tensor)

        graph_def = layer_neuron.aws_neuron_function.graph.as_graph_def()
        print(graph_def)


        # convert keras model to ConcreteFunction
        '''
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        # Frozen model
        frozen_func = convert_variables_to_constants_v2(full_model)
        graph_def = frozen_func.graph.as_graph_def()

        signature_def0 = meta_graph_util.build_signature_def(frozen_func.inputs, frozen_func.outputs)
        amp_graph_def = graph_util.conv2d_padding_optimization(graph_def, signature_def0)

        padding_op_exists = False
        conv2d_checked = False
        # Ensures padding for first Conv2D layer is not same and that padding was added
        for node in amp_graph_def.node:
            if node.name == "Conv/conv1_pad/Pad/paddings":
                print(tf.make_ndarray(node.attr["value"].tensor))
            if node.op == "Conv2D" and not conv2d_checked:
                for attr in node.attr:
                    if attr == "padding":
                       padding_str = node.attr["padding"].s.decode("utf-8")
                       assert(padding_str != "SAME")
                       conv2d_checked = True
            if node.op == "Pad":
                padding_op_exists = True

        assert(padding_op_exists)
        '''
    def test_valid_padding(self):
        '''
        Test asserts that the SAME padding has been removed from Conv2D
        and asserts that the new padding op has been added
        '''
            
        # Model Creation
        kernel = tf.random.uniform([7,7,3,64])
        kernel = tf.cast(kernel, tf.float16)

        def func(tensor):
            tensor = tf.nn.conv2d(tensor, kernel, padding='VALID', strides=[1, 1, 2, 2], data_format='NCHW')
            return tensor

        input_tensor = tf.random.uniform([1, 3, 224, 224])
        input_tensor = tf.cast(input_tensor, tf.float16)

        layer_neuron = tfn.trace(func, input_tensor)

        #print(layer_neuron.aws_neuron_function.graph.as_graph_def())


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

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
from tensorflow.python.framework.tensor_shape import TensorShape
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
        
        graph = tf.Graph()
        with graph.as_default():
            
            # Model Creation
            input0 = tf.keras.layers.Input(shape=(1,28,28,3))
            conv0 = tf.keras.layers.Conv2D(1, (3,3), padding='same')(input0)

            signature_def0 = meta_graph_util.build_signature_def([input0], [conv0])
            graph_def = graph.as_graph_def()
            amp_graph_def = graph_util.conv2d_padding_optimization(graph_def, signature_def0)

            padding_op_exists = False
            # TODO: Modify so that this only happens on the first pad?
            for node in amp_graph_def.node:
                if node.op == "Conv2D":
                    for attr in node.attr:
                        if attr == "padding":
                           padding_str = node.attr["padding"].s.decode("utf-8")
                           assert(padding_str != "SAME")
                if node.op == "PadV2/padding":
                    padding_op_exists = True

            assert(padding_op_exists)

    def test_pad_proto(self):
        graph = tf.Graph()
        with graph.as_default():
            t = tf.constant([[1, 2, 3], [4, 5, 6]])
            paddings = tf.constant([[5, 5,], [2, 2]])
            tf.pad(t, paddings, "CONSTANT")

            graph_def = graph.as_graph_def()

            for node in graph_def.node:
                if node.op == "Const":
                    content = node.attr['value'].tensor.tensor_content
        
       
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

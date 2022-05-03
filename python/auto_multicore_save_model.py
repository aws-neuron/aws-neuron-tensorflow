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

'''
File provides necessary modifications to allow modifications to tf-serving for
automatic multicore inference. We assume that the model being passed here has been
compiled with Inferentia. 

Currently testing on TF2.x
TODO: TF1.x support
'''

import argparse
from tensorflow.python import saved_model
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow_neuron.python.graph_util import _neuron_ops

tNeuronOp = 'NeuronOp'

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model_dir', type=str,
                    help='Model Directory of Inferentia compiled model')
args = parser.parse_args()

def convert_model():
    model_dir = args.model_dir
    
    model = saved_model.load(model_dir)
    cfunc = model.aws_neuron_function
    graph_def = cfunc.graph.as_graph_def()

    neuron_ops = [op for op in _neuron_ops(cfunc.graph)]

    for op in neuron_ops:
        op._set_attr('auto_multicore', attr_value_pb2.AttrValue(s=bytes("true", encoding='utf-8')))
        break

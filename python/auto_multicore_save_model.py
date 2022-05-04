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
from copy import deepcopy
import sys
from tensorflow.python import saved_model
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow_neuron.python.graph_util import _neuron_ops
from tensorflow_neuron.python._trace import _wrap_variable_graph_def_as_concrete_function

tNeuronOp = 'NeuronOp'


def add_attr_to_model(arguments):
    '''
    Adds an attribute _automatic_multicore to the model
    '''
    parser = argparse.ArgumentParser(description='CLI for Inferentia Automatic Multicore Inference')
    parser.add_argument('model_dir', type=str,
                        help='Model Directory of Inferentia compiled model')
    parser.add_argument('--new_model_dir', default='new_model', type=str,
                        help='New model directory to save modified graph')
                        
    args = parser.parse_args(arguments)

    # Load model and process signature defs
    model_dir = args.model_dir
    new_model_dir = args.new_model_dir
    
    model = saved_model.load(model_dir)
    func = model.aws_neuron_function
    graph_def = func.graph.as_graph_def()

    saved_model_proto = parse_saved_model(model_dir)
    signature_def = saved_model_proto.meta_graphs[0].signature_def
    signature_def_key_default = saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    if len(signature_def) == 1:
        signature_def_key = list(signature_def.keys())[0]
    elif signature_def_key_default in signature_def:
        signature_def_key = signature_def_key_default
        logging.warning('Selecting default SignatureDef "{}" when loading SavedModel {}'.format(signature_def_key, model_dir))
    else:
        raise NotImplementedError('Not yet supporting SavedModel {} with multiple signatures')


    # Modify graph def to add a new attribute
    new_nodes = []
    for node in graph_def.node:
        if node.op == tNeuronOp:
            copyNode = deepcopy(node)
            newAttrValue = attr_value_pb2.AttrValue(b=True)
            copyNode.attr['_automatic_multicore'].CopyFrom(newAttrValue)
            new_nodes.append(copyNode)
        else:
            new_nodes.append(node)

    mod_graph_def = graph_pb2.GraphDef()
    mod_graph_def.node.extend(new_nodes)

    cfunc = _wrap_variable_graph_def_as_concrete_function(mod_graph_def, func)
    signatures = {signature_def_key: cfunc}

    saved_model.save(model, new_model_dir, signatures)

def convert_model():
    '''
    Reference that parses the args
    '''

    add_attr_to_model(sys.argv[1:])

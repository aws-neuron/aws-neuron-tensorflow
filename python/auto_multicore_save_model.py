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
from tensorflow import Graph, import_graph_def
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops
from tensorflow.python import saved_model
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import attr_value_pb2, graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow_neuron.python._trace import _wrap_variable_graph_def_as_concrete_function
from tensorflow.neuron.python.saved_model import _normalize_tags, _get_signature_def, simple_save
from tensorflow.neuron.python.graph_util import tag_multicore, _graph_def_to_graph

def add_attr_to_model(arguments):
    '''
    Adds an attribute _automatic_multicore to the model
    '''
    parser = argparse.ArgumentParser(description='CLI for Inferentia Automatic Multicore Inference')
    parser.add_argument('model_dir', type=str,
                        help='Model Directory of Inferentia compiled model')
    parser.add_argument('--num_cores', default=1, type=int,
                        help='Number of cores to be replicated with default as 1')
    parser.add_argument('--new_model_dir', default='new_model', type=str,
                        help='New model directory to save modified graph')
                        
    args = parser.parse_args(arguments)

    # Load model and process signature defs
    model_dir = args.model_dir
    num_cores = args.num_cores
    new_model_dir = args.new_model_dir

    config_proto = config_pb2.ConfigProto(allow_soft_placement=True)
    tags = _normalize_tags(None, model_dir)

    is_v2 = False
    with tf_session.Session(graph=ops.Graph(), config=config_proto) as sess:
        meta_graph = saved_model.loader.load.__wrapped__(sess, tags, model_dir)
        for op in sess.graph.get_operations():
            if op.type == 'StatefulPartitionedCall':
                is_v2 = True

    if not is_v2:

        with tf_session.Session(graph=ops.Graph()) as sess:
            saved_model.loader.load(sess, ['serve'], model_dir)
            graph_def = sess.graph.as_graph_def()
            # Modify graph def to add a new attribute
            tag_multicore(graph_def, num_cores)
        mod_graph  = _graph_def_to_graph(graph_def) 

        with tf_session.Session(graph=mod_graph, config=config_proto) as sess:
            builder = saved_model.builder.SavedModelBuilder(new_model_dir)
            signature_def_key, signature_def = _get_signature_def(meta_graph, None)
            signature_def_map = {signature_def_key: signature_def}
            for tensor in signature_def.inputs.values():
                infer_tensor = mod_graph.get_tensor_by_name(tensor.name)
                tensor.tensor_shape.CopyFrom(infer_tensor.shape.as_proto())
            for tensor in signature_def.outputs.values():
                infer_tensor = mod_graph.get_tensor_by_name(tensor.name)
                tensor.tensor_shape.CopyFrom(infer_tensor.shape.as_proto())
            saved_model_main_op = meta_graph.collection_def['saved_model_main_op'].node_list.value
            saved_model_main_op = [sess.graph.get_operation_by_name(name) for name in saved_model_main_op]
            main_op = saved_model_main_op[0] if saved_model_main_op else None
            builder.add_meta_graph_and_variables(sess, tags, signature_def_map=signature_def_map,
                                                 strip_default_attrs=False,
                                                 main_op=main_op)
            builder.save()
    else:
        model = saved_model.load(model_dir)
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
        func = model.aws_neuron_function
        graph_def = func.graph.as_graph_def()

        # Modify graph def to add a new attribute
        mod_graph_def = tag_multicore(graph_def, num_cores)

        cfunc = _wrap_variable_graph_def_as_concrete_function(mod_graph_def, func)
        signatures = {signature_def_key: cfunc}

        model.aws_neuron_function = cfunc

        saved_model.save(model, new_model_dir, signatures)



def convert_model():
    '''
    Reference that parses the args
    '''

    add_attr_to_model(sys.argv[1:])

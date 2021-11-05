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
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import tf_logging as logging
from tensorflow.neuron.python import graph_def_util as gdu
from tensorflow.neuron.python import utils


def build_signature_def(input_tensors, output_tensors):
    sdef = meta_graph_pb2.SignatureDef()
    for tensors, info_map in zip([input_tensors, output_tensors], [sdef.inputs, sdef.outputs]):
        for tensor in tensors:
            tensor_info = info_map[tensor.name]
            tensor_info.name = tensor.name
            tensor_info.dtype = tensor.dtype.as_datatype_enum
            tensor_info.tensor_shape.CopyFrom(tensor.shape.as_proto())
    return sdef


def run_grappler_on_subgraphs(graph_def):
    passes = [
        'constfold',
        'debug_stripper',
        'constfold',
        'pruning',
        'dependency',
        'constfold',
        'remap',
        'constfold',
        'memory',
        'constfold',
        'common_subgraph_elimination',
        'constfold',
        'arithmetic',
        'constfold',
        'loop',
        'constfold',
        'aws_neuron_split_conv2d_same_padding'
    ]
    for node in gdu.get_neuron_nodes(graph_def):
        is_compilable, reason = gdu.neuron_node_is_compilable(node)
        if not is_compilable:
            logging.warning('Not running grappler passes on subgraph {} because {}'.format(node.name, reason))
            continue
        subgraph_def = gdu.get_subgraph_def(node)
        input_names = node.attr[gdu.knInputNames].list.s
        input_dtypes = node.attr[gdu.knInputDtypes].list.type
        input_shapes = node.attr[gdu.knInputShapes].list.shape
        output_names = node.attr[gdu.knOutputNames].list.s
        output_dtypes = node.attr[gdu.knOutputDtypes].list.type
        output_shapes = node.attr[gdu.knOutputShapes].list.shape
        sdef = meta_graph_pb2.SignatureDef()
        zip_inputs = zip(input_names, input_dtypes, input_shapes)
        zip_outputs = zip(output_names, output_dtypes, output_shapes)
        for zip_tensors, info_map in zip([zip_inputs, zip_outputs], [sdef.inputs, sdef.outputs]):
            for name, dtype, shape in zip_tensors:
                tensor_info = info_map[name]
                tensor_info.name = name
                tensor_info.dtype = dtype
                tensor_info.tensor_shape.CopyFrom(shape)
        meta_graph_def = meta_graph_pb2.MetaGraphDef(graph_def=subgraph_def)
        meta_graph_def.signature_def['serving_default'].CopyFrom(sdef)
        opt_config = config_pb2.ConfigProto()
        rewriter_config = opt_config.graph_options.rewrite_options
        rewriter_config.meta_optimizer_iterations = 1
        rewriter_config.optimizers.extend(passes)
        with utils.change_grappler_logging_level_according_to_cc_flags():
            subgraph_def = tf_optimizer.OptimizeGraph(opt_config, meta_graph_def)
        node.attr[gdu.knGraphDef].s = subgraph_def.SerializeToString()
    return graph_def

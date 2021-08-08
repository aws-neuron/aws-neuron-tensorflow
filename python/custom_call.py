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
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.neuron.python import graph_def_util as gdu


targetAwsNeuronErf = 'AwsNeuronErf'
targetAwsNeuronSoftplus = 'AwsNeuronSoftplus'


class CustomCallLowering:

    def __init__(self):
        self.restore_map = {}

    def lower(self, graph_def):
        if not any(get_custom_call_target(node) for node in graph_def.node):
            return graph_def
        graph = ops.Graph()
        with graph.as_default():
            importer.import_graph_def(graph_def, name='')
        name_to_op = {op.name: op for op in graph.get_operations()}
        for node in graph_def.node:
            custom_call_target = get_custom_call_target(node)
            if custom_call_target:
                self.restore_map[node.name] = node_def_pb2.NodeDef()
                self.restore_map[node.name].CopyFrom(node)
                graph_op = name_to_op[node.name]
                input_dtypes = [ts.dtype.as_datatype_enum for ts in graph_op.inputs]
                output_dtypes = [ts.dtype.as_datatype_enum for ts in graph_op.outputs]
                inferred_shapes = node.attr.pop(gdu.kNeuronInferredShapes)
                backend_config = attr_value_pb2.NameAttrList()
                backend_config.attr.MergeFrom(node.attr)
                node.attr.clear()
                node.op = '_AwsNeuronCustomOp'
                node.attr['custom_call_target'].s = custom_call_target.encode()
                node.attr['backend_config'].s = backend_config.SerializeToString()
                node.attr['input_dtypes'].list.type[:] = input_dtypes
                node.attr['output_dtypes'].list.type[:] = output_dtypes
                node.attr['output_shapes'].CopyFrom(inferred_shapes)
                node.attr[gdu.kNeuronInferredShapes].CopyFrom(inferred_shapes)
        return graph_def

    def restore(self, graph_def):
        for node in graph_def.node:
            if node.name in self.restore_map:
                node.CopyFrom(self.restore_map[node.name])
        return graph_def


def get_custom_call_target(node):
    if node.op == 'Erf':
        return targetAwsNeuronErf
    elif node.op == 'Softplus':
        return targetAwsNeuronSoftplus
    return ''

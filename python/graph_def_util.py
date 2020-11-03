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
from tensorflow.core.framework import graph_pb2


_NEURON_OP = 'NeuronOp'


def restore_compiler_failures(compiled_graph_def, graph):
    """Restore `NeuronOp`'s that failed to compile
    """
    neuron_op_dict = {node.name: node for node in compiled_graph_def.node if node.op == _NEURON_OP}
    restore_nodes = []
    remove_node_names = set()
    gd_tensor_name_map = {}
    for node in get_neuron_nodes(compiled_graph_def):
        if not node.attr['executable'].s:
            remove_node_names.add(node.name)
            subgraph_def = get_subgraph_def(node)
            sgd_tensor_name_map = {}
            for gd_ts_name, sg_ph_name in zip(node.input, node.attr['input_names'].list.s):
                sgd_ph_name = format_tensor_name(sg_ph_name.decode())
                op_name, ts_index = _graph_def_op_index(gd_ts_name)
                if op_name in neuron_op_dict:
                    in_node = neuron_op_dict[op_name]
                    if not in_node.attr['executable'].s:
                        gd_ts_name = in_node.attr['output_names'].list.s[ts_index].decode()
                sgd_tensor_name_map[sgd_ph_name] = gd_ts_name
            for sg_node in subgraph_def.node:
                for idx, name in enumerate(sg_node.input):
                    sg_node.input[idx] = sgd_tensor_name_map.get(name, name)
                if sg_node.op != 'Placeholder':
                    restore_nodes.append(sg_node)
            for out_idx, out_name in enumerate(node.attr['output_names'].list.s):
                out_gd_ts_name = format_tensor_name('{}:{}'.format(node.name, out_idx))
                gd_tensor_name_map[out_gd_ts_name] = format_tensor_name(out_name.decode())
    restore_node_names = {node.name for node in restore_nodes}
    remove_node_names.update(
        node.name for node in compiled_graph_def.node if node.name in restore_node_names)
    for node in restore_nodes:
        op_original = graph.get_operation_by_name(node.name)
        for control_input in op_original.control_inputs:
            node.input.append('^{}'.format(control_input.name))
    for node in compiled_graph_def.node:
        for idx, name in enumerate(node.input):
            node.input[idx] = gd_tensor_name_map.get(name, name)

    graph_def = graph_pb2.GraphDef()
    graph_def.node.extend(
        node for node in compiled_graph_def.node if node.name not in remove_node_names)
    graph_def.node.extend(node for node in restore_nodes)
    return graph_def


def get_neuron_nodes(graph_def):
    return (node for node in graph_def.node if node.op == _NEURON_OP)


def get_subgraph_def(node):
    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(node.attr['graph_def'].s)
    return graph_def


def format_tensor_name(tensor_name):
    return tensor_name.split(':')[0] if tensor_name.endswith(':0') else tensor_name


def _graph_def_op_index(graph_def_tensor_name):
    comma_split = graph_def_tensor_name.split(':')
    if ':' in graph_def_tensor_name:
        op_name, value_index = comma_split
        value_index = int(value_index)
    else:
        op_name, value_index = comma_split[0], 0
    return op_name, value_index

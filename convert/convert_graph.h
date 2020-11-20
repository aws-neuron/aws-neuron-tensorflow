/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_NEURON_CONVERT_GRAPH_H_
#define TENSORFLOW_NEURON_CONVERT_GRAPH_H_

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace neuron {
namespace convert {

struct SubGraphParams {
  SubGraphParams(tensorflow::Graph &inp_graph,
                 const std::set<int> &subgraph_node_id_numbers,
                 const std::vector<string> &output_tensor_names,
                 const std::vector<std::pair<int, int> > &output_indices,
                 tensorflow::EdgeSet &incoming_edges,
                 tensorflow::Node *constructed_neuron_node, const int neuron_op_ind,
                 std::unordered_map<string, int> *neuron_op_index_to_name_map)
      : graph(&inp_graph),
        output_names(&output_tensor_names),
        subgraph_node_ids(&subgraph_node_id_numbers),
        output_inds(&output_indices),
        neuron_node(constructed_neuron_node),
        neuron_op_index(neuron_op_ind),
        neuron_op_index_to_name_map(neuron_op_index_to_name_map) {
    for (const tensorflow::Edge *edge : incoming_edges) {
      subgraph_incoming_edges.push_back(edge);
    }
  }

  tensorflow::Graph *graph;
  const std::vector<string> *output_names;
  const std::set<int> *subgraph_node_ids;
  const std::vector<std::pair<int, int> > *output_inds; // {node_id, output_idx}
  std::vector<const tensorflow::Edge*> subgraph_incoming_edges;
  tensorflow::Node *neuron_node;
  const int neuron_op_index;
  std::unordered_map<string, int> *neuron_op_index_to_name_map;
};

struct ConvertGraphParams {
  ConvertGraphParams(tensorflow::Graph &inp_graph,
                     const std::vector<string> &output_tensor_names,
                     const std::set<int> &subgraph_node_id_numbers,
                     int neuron_op_index,
                     std::unordered_map<string, int> *neuron_op_index_to_name_map)
      : graph(&inp_graph),
        output_names(&output_tensor_names),
        subgraph_node_ids(&subgraph_node_id_numbers),
        neuron_op_index_to_name_map(neuron_op_index_to_name_map),
        neuron_op_count(neuron_op_index) {}

  tensorflow::Graph *graph;
  const std::vector<string> *output_names;
  const std::set<int> *subgraph_node_ids;
  std::vector<std::pair<int, int> > subgraph_outputs;
  tensorflow::EdgeSet subgraph_incoming_edges;
  tensorflow::EdgeSet subgraph_outgoing_edges;
  std::unordered_map<string, int> *neuron_op_index_to_name_map;
  int neuron_op_count;
};

Status CreateNeuronGraphDef(GraphDef *new_graph_def,
                            const GraphDef &graph_def,
                            const std::vector<std::string> &inputs,
                            const std::vector<std::string> &outputs,
                            const int minimum_segment_size,
                            const std::set<std::string> &op_whitelist,
                            const std::set<std::string> &no_fuse_ops,
                            const std::set<std::string> &force_fuse_ops);

}  // namespace convert
}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_CONVERT_GRAPH_H_

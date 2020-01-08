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

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace neuron {
namespace convert {

const int FP32MODE = 0;
const int FP16MODE = 1;
const int INT8MODE = 2;

struct SubGraphParams {
  SubGraphParams(tensorflow::Graph &inp_graph,
                 const std::set<int> &subgraph_node_id_numbers,
                 const std::vector<string> &output_node_names,
                 const std::vector<std::pair<int, int> > &output_indices,
                 tensorflow::EdgeSet &incoming_edges,
                 tensorflow::Node *constructed_eia_node, const int eop_ind,
                 std::unordered_map<string, int> *eop_index_to_name_map,
                 string model_id, int engine_precision_mode = FP32MODE)
      : graph(inp_graph),
        subgraph_node_ids(subgraph_node_id_numbers),
        output_names(output_node_names),
        output_inds(output_indices),
        eia_node(constructed_eia_node),
        eop_index(eop_ind),
        eop_index_to_name_map(eop_index_to_name_map),
        p_model_id(model_id),
        precision_mode(engine_precision_mode) {
    for (const tensorflow::Edge *edge : incoming_edges) {
      subgraph_incoming_edges.push_back(edge);
    }
  }

  tensorflow::Graph &graph;
  const std::vector<string> &output_names;
  const std::set<int> &subgraph_node_ids;
  const std::vector<std::pair<int, int> > &output_inds; // {node_id, output_idx}
  std::vector<const tensorflow::Edge*> subgraph_incoming_edges;
  tensorflow::Node *eia_node;
  const int eop_index;
  const int precision_mode;
  std::unordered_map<string, int> *eop_index_to_name_map;
  const string p_model_id;
};

// TODO(sami): convert references to pointers
struct ConvertGraphParams {
  ConvertGraphParams(tensorflow::Graph &inp_graph,
                     const std::vector<string> &output_node_names,
                     const std::set<int> &subgraph_node_id_numbers,
                     int engine_precision_mode, int eop_index,
                     std::unordered_map<string, int> *eop_index_to_name_map,
                     string model_id)
      : graph(inp_graph),
        output_names(output_node_names),
        subgraph_node_ids(subgraph_node_id_numbers),
        precision_mode(engine_precision_mode),
        eop_index_to_name_map(eop_index_to_name_map),
        eop_count(eop_index),
        p_model_id(model_id) {}

  tensorflow::Graph &graph;
  const std::vector<string> &output_names;
  const std::set<int> &subgraph_node_ids;
  int precision_mode;
  std::vector<std::pair<int, int> > subgraph_inputs;
  std::vector<std::pair<int, int> > subgraph_outputs;
  tensorflow::EdgeSet subgraph_incoming_edges;
  tensorflow::EdgeSet subgraph_outgoing_edges;
  std::unordered_map<string, int> *eop_index_to_name_map;
  int eop_count;
  const string p_model_id;
};

Status ConvertGraphDefToEIA(string *new_graph_def, const string &graph_def,
                            const string &inputs, const string &outputs,
                            const string &op_whitelist, const string &no_fuse_ops,
                            const string &force_fuse_ops, const int min_seg_size);

}  // namespace convert
}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_CONVERT_GRAPH_H_

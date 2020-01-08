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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/compiler/tf2tensorrt/segment/segment.h"
#include "tensorflow/python/neuron/convert/convert_graph.h"


namespace tensorflow {
namespace neuron {
namespace convert {


// Copied from tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h
// Helper class for the segmenter to determine whether an output edge from the
// TRT segment is valid.
class OutputEdgeValidator {
 public:
  // Return true if the specified edge is eligible to be an output edge of the
  // TRT segment.
  bool operator()(const Edge* out_edge) const {
    if (out_edge->IsControlEdge()) return true;
    if (out_edge->src()->type_string() == "Const") {
      VLOG(1) << "--> Need to remove output node " << out_edge->src()->name()
              << " which is a Const.";
      return false;
    }
    return true;
  }
};


// Helper function to split tensor and tensor spec (i.e conv:2 -> conv, 2)
std::vector<string> split(string str, string token) {
  std::vector<string> result;
  while (str.size()) {
    size_t index = str.find(token);
    if (index != string::npos) {
      result.push_back(str.substr(0, index));
      str = str.substr(index + token.size());
      if (str.size() == 0) result.push_back(str);
    } else {
      result.push_back(str);
      str = "";
    }
  }
  return result;
}

// Helper function
void GetSubGraphIncomingEdges(const tensorflow::Graph& graph,
                              const std::set<int>& subgraph_node_ids,
                              tensorflow::EdgeSet* incoming_edges) {
  for (int node_id : subgraph_node_ids) {
    const tensorflow::Node* node = graph.FindNodeId(node_id);
    for (const tensorflow::Edge* edge : node->in_edges()) {
      if (!subgraph_node_ids.count(edge->src()->id()) &&
          !edge->src()->IsSource()) {
        VLOG(2) << "src: " << edge->src()->name()
                << " dst: " << edge->dst()->name();
        incoming_edges->insert(edge);
      } else {
        VLOG(2) << node->name() << " -> " << edge->src()->name() << " N, ";
      }
    }
  }
}

// Helper function
void GetSubGraphOutgoingEdges(const tensorflow::Graph& graph,
                              const std::set<int>& subgraph_node_ids,
                              tensorflow::EdgeSet* outgoing_edges) {
  for (int node_id : subgraph_node_ids) {
    const tensorflow::Node* node = graph.FindNodeId(node_id);
    for (const tensorflow::Edge* edge : node->out_edges()) {
      if (!subgraph_node_ids.count(edge->dst()->id()) &&
          !edge->dst()->IsSink()) {
        VLOG(2) << node->name() << " -> " << edge->dst()->name() << " Y, ";
        outgoing_edges->insert(edge);
      } else {
        VLOG(2) << node->name() << " -> " << edge->dst()->name() << " N, ";
      }
    }
  }
}

// Helper function
std::pair<string, int> ParseTensorName(const string& name,
                                       int default_idx = 0) {
  string name_no_idx = name;
  int idx = default_idx;
  const size_t sep = name_no_idx.find_last_of(':');
  if (sep != string::npos) {
    name_no_idx = name_no_idx.substr(0, sep);
    idx = std::stoi(name.substr(sep + 1));
  }
  return std::make_pair(name_no_idx, idx);
}

tensorflow::Status BuildNodeMap(
    const tensorflow::Graph& graph,
    std::unordered_map<string, tensorflow::Node*>* node_map) {
  for (auto* node : graph.op_nodes()) {
    if (!node_map->insert({node->name(), node}).second) {
      return tensorflow::errors::AlreadyExists(
          "Node name is not unique in graph: " + node->name());
    }
  }
  return tensorflow::Status::OK();
}

// Helper function to return tensor_name and index in a vector
// will return 0 if name doesn't contain ':' in name.
std::vector<string> split_tensor(string out_tensor) {
  std::vector<string> result;
  size_t found = out_tensor.find_last_of(":");

  result.push_back(out_tensor.substr(0, found));
  auto index = (found == string::npos) ? "0" : out_tensor.substr(found + 1);
  result.push_back(index);

  return result;
}

// This function creates subgraph graph def and adds to main graph.
tensorflow::Status ConvertSubGraphToNeuronNodeDef(SubGraphParams &sg_params) {
  string neuron_op_name = tensorflow::strings::StrCat("neuron_op_", sg_params.neuron_op_index);
  VLOG(1) << "Start Node building ...." << neuron_op_name;

  std::vector<string> input_names;
  std::vector<tensorflow::DataType> input_dtypes;
  std::vector<tensorflow::PartialTensorShape> input_shapes;
  std::vector<NodeBuilder::NodeOut> input_nodes;

  // map of nodename -> nodef of nodes to be added subgraph graphdef
  std::unordered_map<std::string, NodeDef> subgraph_nodes;

  // map of (name ->  vector of datatype)
  // This is to store all the nodes in subgraph that is named as output_nodes
  // for the graph
  std::map<string, std::vector<DataType> > main_graph_output_nodes;
  std::map<string, std::vector<PartialTensorShape> > main_graph_output_shapes;

  // map of output_name -> max tensor index,  transforming list of output_names
  // of graph to map i.e list output_names : A:1, A: 3, B:0 => map: A->3, B->0
  std::unordered_map<string, int> map_out_names_to_index;

  for (auto out_tensor : *sg_params.output_names) {
    auto tensor_vec = split_tensor(out_tensor);
    map_out_names_to_index[tensor_vec[0]] = std::max(
        map_out_names_to_index[tensor_vec[0]], std::stoi(tensor_vec[1]));
  }

  // Adding placeholder for all incoming edges, since
  // placeholders do not need to specify inputs.
  for (auto incoming_edge : sg_params.subgraph_incoming_edges) {
    if (incoming_edge->IsControlEdge()) {
      VLOG(2) << "Control edge: src-> " << incoming_edge->src()->name()
              << " dst-> " << incoming_edge->dst()->name();

      // Need to remove this input from the node. Control edges incoming to
      // the subgraph should point to Neuron op directly. The inputs in nodes
      // taking control edges from outside should be removed.
      NodeDef inside_node_def;
      std::vector<string> input_list;
      auto inside_node_name = incoming_edge->dst()->name();

      if (subgraph_nodes.find(inside_node_name) == subgraph_nodes.end()) {
        inside_node_def = incoming_edge->dst()->def();
      } else {
        inside_node_def = subgraph_nodes[inside_node_name];
      }

      for (int i = 0; i < inside_node_def.input_size(); ++i) {
        auto input_name = inside_node_def.input(i);
        if (input_name[0] != '^') {
          input_list.push_back(input_name);
        }
      }

      // Clearing the input and adding back all the inputs except
      // control inputs.
      inside_node_def.clear_input();
      for (auto input : input_list) {
        inside_node_def.add_input(input);
      }

      // adding the new nodedef to the list.
      subgraph_nodes[incoming_edge->dst()->name()] = inside_node_def;
      continue;
    }

    int outside_node_id = incoming_edge->src()->id();
    int outside_node_index = incoming_edge->src_output();
    int inside_node_id = incoming_edge->dst()->id();
    int inside_node_index = incoming_edge->dst_input();

    tensorflow::Node* inside_node = sg_params.graph->FindNodeId(inside_node_id);
    tensorflow::Node* outside_node = sg_params.graph->FindNodeId(outside_node_id);

    VLOG(2) << "edge->  src:" << outside_node->name()
            << " idx: " << outside_node_index << " dst:" << inside_node->name()
            << " idx: " << inside_node_index;

    NodeDef placeholder_def;
    auto placeholder_name = sg_params.graph->NewName(
      outside_node->name() + std::to_string(outside_node_index));
    std::vector<PartialTensorShape> v_partial_shape;
    input_dtypes.push_back(outside_node->output_type(outside_node_index));
    if (GetNodeAttr(outside_node->attrs(), "_output_shapes", &v_partial_shape).ok()) {
        input_shapes.push_back(v_partial_shape[outside_node_index]);
        TF_RETURN_IF_ERROR(
            NodeDefBuilder(placeholder_name, "Placeholder")
                .Attr("shape", v_partial_shape[outside_node_index])
                .Attr("_output_shapes", v_partial_shape[outside_node_index])
                .Attr("dtype", outside_node->output_type(outside_node_index))
                .Finalize(&placeholder_def));
    } else {
        input_shapes.emplace_back();
        TF_RETURN_IF_ERROR(
            NodeDefBuilder(placeholder_name, "Placeholder")
                .Attr("dtype", outside_node->output_type(outside_node_index))
                .Finalize(&placeholder_def));
    }

    // Changing the inside nodes taking inputs from outside to point to
    // placeholder created above. If Nodedef of the node is already added to
    // the map, update the same nodedef, else create a nodedef from the node.
    NodeDef inside_node_def =
        subgraph_nodes.find(inside_node->name()) != subgraph_nodes.end()
            ? subgraph_nodes[inside_node->name()]
            : inside_node->def();
    inside_node_def.set_input(inside_node_index, placeholder_name.c_str());

    VLOG(2) << "Input node def:" << inside_node_def.DebugString();
    VLOG(2) << " Place holder def :" << placeholder_def.DebugString();

    // for creating Neuron op node, storing:
    // input_names -> Attr(input_names) -> name of input nodes inside subgraph
    input_names.push_back(placeholder_name + ":0");

    // input_nodes -> Input(input_nodes) ->. list of nodes with index (outside
    // the subgraph) feeding into the subgraph.
    input_nodes.push_back(
        NodeBuilder::NodeOut(outside_node, outside_node_index));

    // Storing/updating  values in  nodedef map
    subgraph_nodes[placeholder_name] = (placeholder_def);
    subgraph_nodes[inside_node->name()] = (inside_node_def);
  }

  // Debug code
  VLOG(2) << " Incoming nodes in graphdef";
  for (auto it = subgraph_nodes.begin(); it != subgraph_nodes.end(); it++) {
    VLOG(2) << "name:" << it->first << "\n Nodedef:\n"
            << it->second.DebugString();
  }

  // Adding all the interior nodes to the list.
  for (auto node_id : *sg_params.subgraph_node_ids) {
    tensorflow::Node *node = sg_params.graph->FindNodeId(node_id);
    // this is to handle main graph output nodes in this segment.
    // filling map of (tensor_name , vector(dtype)) i.e A ->DT_FLOAT,->INT8
    // dtype order will match the the index.
    if (map_out_names_to_index.count(node->name())) {
      std::vector<PartialTensorShape> v_partial_shape;
      if (!GetNodeAttr(node->attrs(), "_output_shapes", &v_partial_shape).ok()) {
        v_partial_shape.clear();
        for (auto tensor_index = 0;
           tensor_index <= map_out_names_to_index[node->name()];
           ++tensor_index) {
          v_partial_shape.push_back(PartialTensorShape());
        }
      }
      for (auto tensor_index = 0;
           tensor_index <= map_out_names_to_index[node->name()];
           ++tensor_index) {
        DataType tensor_dtype = node->output_type(tensor_index);
        main_graph_output_nodes[node->name()].push_back(tensor_dtype);
        main_graph_output_shapes[node->name()].push_back(v_partial_shape[tensor_index]);
        VLOG(1) << " ConvertSubGraphToNeuronNodeDef node: making pair: ("
                << node->name() << " , " << tensor_dtype << " )";
      }
    }
    if (subgraph_nodes.find(node->name()) == subgraph_nodes.end()) {
      subgraph_nodes[node->name()] = node->def();
    }
  }

  // Inserting all the nodedef to graphdef for subgraph
  GraphDef subgraph_graph_def;
  for (auto it = subgraph_nodes.begin(); it != subgraph_nodes.end(); it++) {
    (*subgraph_graph_def.add_node()) = it->second;
  }

  VLOG(2) << "Neuron subgraph graphdef: " << subgraph_graph_def.DebugString();

  string in_graph_def_string = "";
  if (!subgraph_graph_def.SerializeToString(&in_graph_def_string)) {
    return tensorflow::errors::InvalidArgument("Failed to serialize subgraph");
  }

  // Gather output metadata
  VLOG(2) << "Neuron op: " << neuron_op_name
          << " sg_params.output_inds: " << sg_params.output_inds->size();
  std::vector<string> neuron_node_output_names;
  std::vector<tensorflow::DataType> neuron_node_output_dtypes;
  std::vector<PartialTensorShape> neuron_node_output_shapes;
  for (const std::pair<int, int>& output : *sg_params.output_inds) {
    int node_id = output.first;
    int output_idx = output.second;
    tensorflow::Node* node = sg_params.graph->FindNodeId(node_id);
    string op_name = node->name();
    string tensor_name = op_name;

    VLOG(2) << "op_name: " << op_name;
    tensorflow::strings::StrAppend(&tensor_name, ":", output_idx);
    VLOG(2) << "Output tensor name: " << tensor_name;
    neuron_node_output_names.push_back(tensor_name);

    tensorflow::DataType tf_dtype = node->output_type(output_idx);
    neuron_node_output_dtypes.push_back(tf_dtype);

    std::vector<PartialTensorShape> v_partial_shape;
    if (GetNodeAttr(node->attrs(), "_output_shapes", &v_partial_shape).ok()) {
      neuron_node_output_shapes.push_back(v_partial_shape[output_idx]);
    } else {
      neuron_node_output_shapes.emplace_back();
    }
  }

  // Pushing the outputs of graph (if in the subgraph) to the output for the
  // subgraph. As we will create IdentityN connecting to the these nodes.
  for (auto name_type : main_graph_output_nodes) {
    uint counter = 0;
    auto name = name_type.first;
    auto dtype = name_type.second;
    for (auto dtype : name_type.second) {
      VLOG(1) << "tensorname: " << name << " dtype: " << dtype;
      neuron_node_output_names.push_back(name + ":" + std::to_string(counter++));
      neuron_node_output_dtypes.push_back(dtype);
    }
    for (auto shape : main_graph_output_shapes[name]) {
      neuron_node_output_shapes.push_back(shape);
    }
  }

  VLOG(1) << "Finished op preparation";

  StringPieceHasher hasher;
  std::string hash_string = "";
  for (auto const& s : input_names) {
    hash_string += s;
  }
  for (auto const& s : neuron_node_output_names) {
    hash_string += s;
  }
  for (auto const& s : neuron_node_output_dtypes) {
    hash_string += s;
  }
  std::stringstream stream;
  stream << std::hex << hasher(hash_string);
  std::string hex_string(stream.str());

  VLOG(2) << "String to be hashed " << hash_string << "Hashed String "
          << hasher(hash_string) << "Hex Rep" << hex_string;

  neuron_op_name = "neuron_op_" + hex_string;
  sg_params.neuron_op_index_to_name_map->insert({hex_string, sg_params.neuron_op_index});

  Node *neuron_node;
  TF_CHECK_OK(NodeBuilder(neuron_op_name, "NeuronOp")
                  .Input(input_nodes)
                  .Attr("graph_def", in_graph_def_string)
                  .Attr("input_names", input_names)
                  .Attr("input_dtypes", input_dtypes)
                  .Attr("input_shapes", input_shapes)
                  .Attr("output_names", neuron_node_output_names)
                  .Attr("output_dtypes", neuron_node_output_dtypes)
                  .Attr("output_shapes", neuron_node_output_shapes)
                  .Attr("_output_shapes", neuron_node_output_shapes)
                  .Finalize(sg_params.graph, &neuron_node));

  sg_params.neuron_node = neuron_node;

  // Creating identityN node corresponding to any output nodes(if any) in this
  // subgraph
  // start_index is the index of Neuron op output to which IdentityN node should be
  // connected to.
  auto start_index = sg_params.output_inds->size();
  VLOG(1) << "start_index: " << start_index;
  std::vector<tensorflow::NodeBuilder::NodeOut> identity_inputs;

  // Iterate through the output node list found.
  for (auto name_index : main_graph_output_nodes) {
    identity_inputs.clear();
    auto name = name_index.first;
    VLOG(1) << " indentity inputs: name:" << name;
    VLOG(1) << " max index: " << map_out_names_to_index[name];
    for (size_t i = 0; i < main_graph_output_nodes[name].size(); ++i) {
      VLOG(1) << "start_index: " << start_index;
      identity_inputs.push_back(NodeBuilder::NodeOut(neuron_node, start_index++));
    }
    Node* node;
    TF_CHECK_OK(NodeBuilder(name, "IdentityN")
                    .Input(identity_inputs)
                    .Finalize(sg_params.graph, &node));
    VLOG(1) << " New output IdentityN node: " << node->def().DebugString();
  }

  VLOG(1) << "Created new node ..............";
  VLOG(2) << " new node: " << neuron_node->def().DebugString();

  return tensorflow::Status::OK();
}

std::unordered_map<string, std::vector<int>> BuildTensorNameMap(
    const std::vector<string>& tensor_names) {
  std::unordered_map<string, std::vector<int>> result;
  for (string const& tensor_name : tensor_names) {
    string node_name;
    int index;
    std::tie(node_name, index) = ParseTensorName(tensor_name);
    VLOG(2) << "node_name : " << node_name << " index:" << index;
    result[node_name].push_back(index);
  }
  return result;
}

// This fills the ConvertGraphParams struct.
static tensorflow::Status FillSubGraphEdgeSets(ConvertGraphParams *params) {
  GetSubGraphIncomingEdges(*params->graph, *params->subgraph_node_ids,
                           &params->subgraph_incoming_edges);

  auto output_name_to_index_map = BuildTensorNameMap(*params->output_names);
  std::set<std::pair<int, int>> subgraph_outputs_set;
  // Collect outputs referenced from output_names
  GetSubGraphOutgoingEdges(*params->graph, *params->subgraph_node_ids,
                           &params->subgraph_outgoing_edges);
  for (const tensorflow::Edge* edge : params->subgraph_outgoing_edges) {
    if (!edge->IsControlEdge())
      subgraph_outputs_set.insert({edge->src()->id(), edge->src_output()});
  }
  params->subgraph_outputs.reserve(subgraph_outputs_set.size());
  params->subgraph_outputs.insert(params->subgraph_outputs.begin(),
                                  subgraph_outputs_set.begin(),
                                  subgraph_outputs_set.end());
  return tensorflow::Status::OK();
}

// Sets up structs for creating subgraph.
// Also, rewires new Neuron node to the graph. Removes old nodes.
tensorflow::Status ConvertSubGraphToNeuron(ConvertGraphParams *params) {
  TF_RETURN_IF_ERROR(FillSubGraphEdgeSets(params));
  tensorflow::NodeDef neuron_node_def;
  tensorflow::Node *neuron_node = nullptr;

  SubGraphParams sg_params(*params->graph, *params->subgraph_node_ids,
                           *params->output_names, params->subgraph_outputs,
                           params->subgraph_incoming_edges, neuron_node,
                           params->neuron_op_count,
                           params->neuron_op_index_to_name_map);

  TF_RETURN_IF_ERROR(ConvertSubGraphToNeuronNodeDef(sg_params));

  tensorflow::Status status;
  neuron_node_def = sg_params.neuron_node->def();
  neuron_node = sg_params.neuron_node;

  // AddNode does not wire edges.
  // Re-map incoming edges to use the new Neuron node instead of the orig subgraph
  std::map<std::pair<int, int>, int> subgraph_edge_to_input_map;
  VLOG(2) << "subgraph_edge_to_input_map: ";
  for (size_t i = 0; i < params->subgraph_inputs.size(); ++i) {
    VLOG(2) << params->subgraph_inputs.at(i).first << " , "
            << params->subgraph_inputs.at(i).second << " i " << i;
    subgraph_edge_to_input_map.insert({params->subgraph_inputs.at(i), i});
  }
  for (const tensorflow::Edge *edge : params->subgraph_incoming_edges) {
    // std::pair<int, int> old_src = {edge->src()->id(), edge->src_output()};
    // int new_src_output = subgraph_edge_to_input_map.at(old_src);
    if (edge->IsControlEdge()) {
      params->graph->AddControlEdge(edge->src(), neuron_node);
    }
    params->graph->RemoveEdge(edge);
  }

  VLOG(2) << "new wiring edges: " << neuron_node->in_edges().size();
  for (const tensorflow::Edge *edge : neuron_node->in_edges()) {
    VLOG(2) << edge->src()->name() << " port: " << edge->src_output();
  }

  TF_RETURN_IF_ERROR(status);

  // Re-map outgoing edges to use the new Neuron node instead of the orig subgraph
  std::map<std::pair<int, int>, int> subgraph_edge_to_output_map;
  for (size_t i = 0; i < params->subgraph_outputs.size(); ++i) {
    subgraph_edge_to_output_map.insert({params->subgraph_outputs.at(i), i});
  }
  TF_RETURN_IF_ERROR(status);
  for (const tensorflow::Edge *edge : params->subgraph_outgoing_edges) {
    if (edge->IsControlEdge()) {
      params->graph->AddControlEdge(neuron_node, edge->dst());
    } else {
      std::pair<int, int> old_src = {edge->src()->id(), edge->src_output()};
      int new_src_output = subgraph_edge_to_output_map.at(old_src);
      TF_RETURN_IF_ERROR(params->graph->UpdateEdge(
          neuron_node, new_src_output, edge->dst(), edge->dst_input()));
    }
  }

  // Remove the original subgraph
  for (int node_id : *params->subgraph_node_ids) {
    tensorflow::Node* node = params->graph->FindNodeId(node_id);
    // Don't remove the input placeholders
    if (node->type_string() == "Placeholder") {
      continue;
    }
    params->graph->RemoveNode(node);
  }
  return tensorflow::Status::OK();
}

static tensorflow::Status ExcludeInputNodes(
    tensorflow::Node* omit_node, std::set<string>& exclude_node_list) {
  for (auto in_edge : omit_node->in_edges()) {
    tensorflow::Node* in_node = in_edge->src();
    exclude_node_list.insert(in_node->name());
    ExcludeInputNodes(in_node, exclude_node_list);
  }

  return tensorflow::Status::OK();
}

// Takes all the segment and modifies input graph to have Neuron ops.
static tensorflow::Status ProcessSegments(
    tensorflow::Graph &graph, const std::vector<string> &output_names,
    std::unordered_map<string, tensorflow::Node*> &node_map,
    tensorflow::tensorrt::segment::SegmentNodesVector &segments, int &neuron_op_index,
    std::unordered_map<string, int> *neuron_op_index_to_name_map) {
  tensorflow::Status status = tensorflow::Status::OK();

  for (const std::set<const Node*>& subgraph_node_names : segments) {
    std::set<int> subgraph_node_ids;
    std::stringstream oss;
    for (const Node *node : subgraph_node_names) {
      oss << " " << node->name();
      if (node_map.find(node->name()) == node_map.end()) {
        string msg =
            "Failed to find node in the graph while creating "
            "processing segments !!!!";
        VLOG(1) << msg;
        return tensorflow::Status(tensorflow::error::Code::INTERNAL, msg);
      }
      subgraph_node_ids.insert(node_map.at(node->name())->id());
    }
    VLOG(1) << "Subgraph num nodes" << subgraph_node_ids.size();
    VLOG(2) << "Subgraph nodes" << oss.str();

    ConvertGraphParams params(graph, output_names, subgraph_node_ids,
                              neuron_op_index++, neuron_op_index_to_name_map);
    tensorflow::Status status = ConvertSubGraphToNeuron(&params);
    if (status != tensorflow::Status::OK()) {
      LOG(WARNING) << "subgraph conversion error for subgraph_index:"
                   << neuron_op_index - 1 << " due to: \"" << status.ToString()
                   << "\" SKIPPING......( " << subgraph_node_names.size()
                   << " nodes)";
    }
  }

  return status;
}

void PreProcessSegmentsForResources(
    tensorflow::Graph &graph,
    tensorflow::tensorrt::segment::SegmentNodesVector &normal_segments,
    tensorflow::tensorrt::segment::SegmentNodesVector &loop_segments,
    // uint& ops_kept_local,
    std::unordered_map<string, int> *neuron_op_index_to_name_map) {
  /*
  For each segment:
    1. If a resouce is input or output of segment, add segment to remove list.
    2. Replace any Neuron ops that is inside the segment with actual nodes. This is
    happens coz we we are currently doing segmentation on graph that might have
    loops inside Neuron ops.
    3. Remove all segments in remove list from segment list.
  */
  std::unordered_map<string, tensorflow::Node*> node_map;
  BuildNodeMap(graph, &node_map);
  std::set<int> remove_segment_index;
  for (auto idx = 0; idx < (int)normal_segments.size(); idx++) {
    std::set<const Node*> &nodes_in_segment = normal_segments[idx];
    std::set<int> subgraph_node_ids;
    for (auto node : nodes_in_segment) {
      subgraph_node_ids.insert(node_map[node->name()]->id());
    }

    tensorflow::EdgeSet incoming_edges;
    tensorflow::EdgeSet outgoing_edges;
    GetSubGraphIncomingEdges(graph, subgraph_node_ids, &incoming_edges);
    GetSubGraphOutgoingEdges(graph, subgraph_node_ids, &outgoing_edges);

    for (const tensorflow::Edge *edge : incoming_edges) {
      if (edge->dst()->input_type(edge->dst_input()) == DT_RESOURCE) {
        VLOG(1) << "incoming edge src: " << edge->src()->name()
                << " dst: " << edge->dst()->name();
        remove_segment_index.insert(idx);
        break;
      }
    }

    for (const tensorflow::Edge *edge : outgoing_edges) {
      if (edge->src()->output_type(edge->src_output()) == DT_RESOURCE) {
        VLOG(1) << "outgoing edge src: " << edge->src()->name()
                << " dst: " << edge->dst()->name();
        remove_segment_index.insert(idx);
        break;
      }
    }
  }

  // Removing segment from segemnt list.
  uint remove_segment_count = 0;
  for (auto idx : remove_segment_index) {
    uint rel_idx = idx - remove_segment_count++;
    VLOG(1) << " Removing segment :" << idx
            << " num of segment nodes : " << normal_segments[rel_idx].size();
    normal_segments.erase(normal_segments.begin() + rel_idx);
  }
}

// Performing :
// 1. Clearing device info from all nodes
// 2. Removing _class:loc attr from nodedef
Status PreProcessingGraphDef(const std::vector<string>& output_names,
                             GraphDef& in_graph_def) {
  VLOG(1) << " Creating Identities for all outputs";

  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             in_graph_def.library());
  tensorflow::Graph graph(flib);
  TF_CHECK_OK(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), in_graph_def, &graph));

  std::unordered_map<string, tensorflow::Node*> node_map;
  BuildNodeMap(graph, &node_map);

  for (int idx = 0; idx < in_graph_def.node_size(); idx++) {
    NodeDef* node_def = in_graph_def.mutable_node(idx);
    auto node_name = node_def->name();

    // clearing devie info from nodedef
    node_def->clear_device();

    // Clearing the colocation information in the node_def
    if (node_def->attr().find("_class") != node_def->attr().end()) {
      node_def->mutable_attr()->erase("_class");
    }
  }

  // Debug code
  // string final_out_graph = "/tmp/startgraph.pb";
  // WriteBinaryProto(Env::Default(), final_out_graph, in_graph_def);

  VLOG(2) << in_graph_def.DebugString();

  return tensorflow::Status::OK();
}

// This function is the base function which does:
// Step 1: Find Neuron Segments.
// Step 2: Calls functions to create Neuron subgraphs.
static tensorflow::Status BuildNeuronOp(
    GraphDef& in_graph_def, const std::vector<string>& inputs,
    const std::vector<string>& tensor_output_names,
    tensorflow::GraphDef& new_graph_def, const int minimum_segment_size,
    std::set<std::string>* op_whitelist,
    std::set<std::string>* no_fuse_ops, std::set<std::string>* force_fuse_ops) {

  // For storing just names without tensor
  std::vector<string> output_names;
  for (auto name : tensor_output_names) {
    output_names.push_back(split(name, ":")[0]);
  }

  // Segment the graph into subgraphs that can be converted to Neuron op
  tensorflow::tensorrt::segment::SegmentOptions segment_options;

  VLOG(1) << "Building Neuron Op\n";

  // Creating Identity node for all output nodes.
  TF_RETURN_IF_ERROR(PreProcessingGraphDef(tensor_output_names, in_graph_def));

  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             in_graph_def.library());
  tensorflow::Graph graph(flib);

  TF_CHECK_OK(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), in_graph_def, &graph));

  std::unordered_map<string, tensorflow::Node*> node_map;
  TF_RETURN_IF_ERROR(BuildNodeMap(graph, &node_map));

  segment_options.minimum_segment_size = minimum_segment_size;

  std::unordered_map<string, int> neuron_op_index_to_name_map;

  // ############################ STEP 1: #################################
  // Making a graph with  with only loops in Neuron ops.

  tensorflow::tensorrt::segment::SegmentNodesVector loop_segments;

  tensorflow::Graph loop_graph(flib);

  TF_CHECK_OK(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), in_graph_def, &loop_graph));
  int start_neuron_op_index = 0;

  // Only do loops segmentation if all the ops in "loop_ops" list
  // are in whitelist.

  // ###################  STEP 2 #########################################
  /// Once the loops are in the graph , we will use that graph to form normal
  /// segments. We will add NeuronOp to whitelist , so that they are engulfed if
  /// needed.

  // Setup exclude_node_list
  for (int i = 0; i < loop_graph.num_node_ids(); ++i) {
    tensorflow::Node* node = loop_graph.FindNodeId(i);
    bool is_source_or_sink = node->IsSink() || node->IsSource();
    bool in_whitelist = op_whitelist->count(node->type_string());
    bool no_fuse = no_fuse_ops->count(node->name());
    bool force_fuse = force_fuse_ops->count(node->name());
    if (!((in_whitelist && !is_source_or_sink && !no_fuse) || force_fuse)) {
      segment_options.exclude_node_list.insert(node->name());
    }
  }

  // All inout nodes to exclude list
  for (auto omit_node_name : inputs) {
    auto node_name = split(omit_node_name, ":")[0];
    segment_options.exclude_node_list.insert(node_name);

    // Adding all the nodes before the input node to exclude list.
    tensorflow::Node* omit_node = node_map[node_name];
    if (omit_node) {
      ExcludeInputNodes(omit_node, segment_options.exclude_node_list);
    }
  }

  tensorflow::tensorrt::segment::SegmentNodesVector normal_segments;
  GraphDef graph_def_with_loops;
  loop_graph.ToGraphDef(&graph_def_with_loops);

  TF_RETURN_IF_ERROR(tensorflow::tensorrt::segment::SegmentGraph(
    &loop_graph,
    [](const Node* node) { return Status::OK(); },
    [](const Edge* edge) { return true; },
    OutputEdgeValidator(),
    segment_options,
    &normal_segments));
  if (normal_segments.size() > 1) {
    VLOG(1) << "MULTIPLE Neuron candidate conversion: " << normal_segments.size();
  }

  if (normal_segments.size()) {
    PreProcessSegmentsForResources(loop_graph, normal_segments, loop_segments,
                                   &neuron_op_index_to_name_map);
    TF_RETURN_IF_ERROR(ProcessSegments(graph, tensor_output_names, node_map,
                                       normal_segments, start_neuron_op_index,
                                       &neuron_op_index_to_name_map));
  }

  // ################### STEP 2 DONE #######################

  graph.ToGraphDef(&new_graph_def);
  VLOG(2) << "new_graph_def: " << new_graph_def.DebugString();
  return tensorflow::Status::OK();
}

static Status CreateNeuronGraphDef(GraphDef &in_graph_def,
                                   const std::vector<string> &inputs,
                                   const std::vector<string> &outputs,
                                   const int minimum_segment_size,
                                   std::set<std::string> *op_whitelist,
                                   std::set<std::string> *no_fuse_ops,
                                   std::set<std::string> *force_fuse_ops,
                                   GraphDef &new_graph_def) {
  OpRegistryInterface *op_registry = OpRegistry::Global();
  // Add default attributes to all nodes in the graph def, This
  // should solve the index_type problem for faster rcnn
  TF_CHECK_OK(AddDefaultAttrsToGraphDef(&in_graph_def, *op_registry, 0, true));

  return BuildNeuronOp(in_graph_def, inputs, outputs, new_graph_def,
                       minimum_segment_size, op_whitelist, no_fuse_ops, force_fuse_ops);
}

// This is the first function that gets called from python (whitelist_partition)
// linked through swig file whitelist_partition.i
Status ConvertGraphDefToNeuron(string *new_graph_def_str,
                               const string &graph_def_str,
                               const string &inputs_str,
                               const string &outputs_str,
                               const string &op_whitelist_str,
                               const string &no_fuse_ops_str,
                               const string &force_fuse_ops_str,
                               const int min_seg_size) {
  tensorflow::GraphDef graph_def, new_graph_def;
  graph_def.ParseFromString(graph_def_str);
  std::vector<string> inputs;
  std::vector<string> outputs;
  tensorflow::AttrValue::ListValue temp;
  temp.ParseFromString(inputs_str);
  for (const auto& name : temp.s()) {
    inputs.push_back(name);
  }
  temp.ParseFromString(outputs_str);
  for (const auto& name : temp.s()) {
    outputs.push_back(name);
  }

  std::set<std::string> op_whitelist;
  temp.ParseFromString(op_whitelist_str);
  for (const auto& name : temp.s()) {
    op_whitelist.insert(name);
  }
  std::set<std::string> no_fuse_ops;
  temp.ParseFromString(no_fuse_ops_str);
  for (const auto& name : temp.s()) {
    no_fuse_ops.insert(name);
  }
  std::set<std::string> force_fuse_ops;
  temp.ParseFromString(force_fuse_ops_str);
  for (const auto& name : temp.s()) {
    force_fuse_ops.insert(name);
  }
  if (min_seg_size < 1) {
    return tensorflow::errors::InvalidArgument("min_seg_size >= 1 required");
  }

  uint64 start_convert_us = Env::Default()->NowMicros();
  Status status = CreateNeuronGraphDef(graph_def, inputs, outputs, min_seg_size,
                                       &op_whitelist, &no_fuse_ops, &force_fuse_ops,
                                       new_graph_def);
  new_graph_def.SerializeToString(new_graph_def_str);
  uint64 convert_time_us = Env::Default()->NowMicros() - start_convert_us;
  VLOG(1) << "Conversion Took " << convert_time_us / 1000 << "ms\n";
  return status;
}

}  // namespace convert
}  // namespace neuron
}  // namespace tensorflow

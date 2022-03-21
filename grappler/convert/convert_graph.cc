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

#include "tensorflow/neuron/grappler/convert/convert_graph.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/neuron/grappler/convert/segment.h"
#include "tensorflow/neuron/grappler/graph_constructor_wrapper.h"
#include "tensorflow/neuron/runtime/macros.h"

namespace tensorflow {
namespace neuron {
namespace convert {

constexpr char kNeuronInferredShapes[] = "_aws_neuron_inferred_shapes";
constexpr char kNeuronInFixedShapeContext[] =
    "_aws_neuron_in_fixed_shape_context";

class EdgeValidator {
 public:
  // Return true if the specified edge is eligible to be an input edge of the
  // Neuron segment.
  bool operator()(const Edge* edge) const {
    const Node* node = edge->src();
    int port = edge->src_output();
    DataType dtype = node->output_type(port);
    return !illegal_dtypes_.count(dtype);
  }

 private:
  const std::unordered_set<DataType> illegal_dtypes_ = {DT_INT64, DT_DOUBLE};
};

// Helper class for the segmenter to determine whether an output edge from the
// Neuron segment is valid.
class OutputEdgeValidator {
 public:
  // Return true if the specified edge is eligible to be an output edge of the
  // Neuron segment.
  bool operator()(const Edge* out_edge) const {
    if (out_edge->IsControlEdge()) return true;
    if (out_edge->src()->type_string() == "Const") {
      VLOG(3) << "--> Need to remove output node " << out_edge->src()->name()
              << " which is a Const.";
      return false;
    }
    return EdgeValidator()(out_edge);
  }
};

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

static bool IsPlaceholder(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Placeholder" || op == "PlaceholderV2" ||
         op == "PlaceholderWithDefault";
}

// This function creates subgraph graph def and adds to main graph.
tensorflow::Status ConvertSubGraphToNeuronNodeDef(SubGraphParams& sg_params) {
  string neuron_op_name =
      tensorflow::strings::StrCat("neuron_op_", sg_params.neuron_op_index);
  VLOG(3) << "Start Node building ...." << neuron_op_name;

  std::vector<string> input_names;
  std::vector<tensorflow::DataType> input_dtypes;
  std::vector<tensorflow::PartialTensorShape> input_shapes;
  std::vector<NodeBuilder::NodeOut> input_nodes;

  // map of nodename -> nodef of nodes to be added subgraph graphdef
  std::unordered_map<std::string, NodeDef> subgraph_nodes;

  // map of (name ->  vector of datatype)
  // This is to store all the nodes in subgraph that is named as output_nodes
  // for the graph
  std::map<string, std::vector<std::pair<DataType, PartialTensorShape>>>
      main_graph_output_nodes;

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
  std::unordered_map<std::string, std::string>
      outside_name_port_to_placeholder_name;
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
    tensorflow::Node* outside_node =
        sg_params.graph->FindNodeId(outside_node_id);

    VLOG(2) << "edge->  src:" << outside_node->name()
            << " idx: " << outside_node_index << " dst:" << inside_node->name()
            << " idx: " << inside_node_index;

    NodeDef placeholder_def;
    std::string outside_node_port =
        outside_node->name() + std::to_string(outside_node_index);
    std::string placeholder_name;
    if (outside_name_port_to_placeholder_name.count(outside_node_port)) {
      placeholder_name =
          outside_name_port_to_placeholder_name[outside_node_port];
    } else {
      placeholder_name = sg_params.graph->NewName(outside_node_port);
      outside_name_port_to_placeholder_name[outside_node_port] =
          placeholder_name;
      PartialTensorShape out_shape;
      std::vector<PartialTensorShape> v_output_shape;
      std::vector<PartialTensorShape> v_inferred_shape;
      PartialTensorShape inferred_shape;
      AttrSlice attrs = outside_node->attrs();
      DataType out_dtype = outside_node->output_type(outside_node_index);
      if (IsPlaceholder(outside_node->def()) &&
          GetNodeAttr(attrs, "shape", &out_shape).ok()) {
      } else if (GetNodeAttr(attrs, "_output_shapes", &v_output_shape).ok()) {
        out_shape = v_output_shape[outside_node_index];
      }
      if (GetNodeAttr(attrs, kNeuronInferredShapes, &v_inferred_shape).ok()) {
        inferred_shape = v_inferred_shape[outside_node_index];
      } else {
        inferred_shape = out_shape;
      }
      TF_RETURN_IF_ERROR(NodeDefBuilder(placeholder_name, "Placeholder")
                             .Attr("dtype", out_dtype)
                             .Attr("shape", out_shape)
                             .Attr(kNeuronInferredShapes, {inferred_shape})
                             .Finalize(&placeholder_def));

      // for creating Neuron op node, storing:
      // input_names -> Attr(input_names) -> name of input nodes inside subgraph
      // input_dtypes -> Attr(input_dtypes) -> dtype of input nodes inside
      // subgraph input_shapes -> Attr(input_shapes) -> shape of input nodes
      // inside subgraph
      input_names.push_back(placeholder_name + ":0");
      input_dtypes.push_back(out_dtype);
      input_shapes.push_back(inferred_shape);

      // input_nodes -> Input(input_nodes) ->. list of nodes with index (outside
      // the subgraph) feeding into the subgraph.
      input_nodes.push_back(
          NodeBuilder::NodeOut(outside_node, outside_node_index));

      subgraph_nodes[placeholder_name] = placeholder_def;
    }

    // Changing the inside nodes taking inputs from outside to point to
    // placeholder created above. If Nodedef of the node is already added to
    // the map, update the same nodedef, else create a nodedef from the node.
    NodeDef inside_node_def =
        subgraph_nodes.find(inside_node->name()) != subgraph_nodes.end()
            ? subgraph_nodes[inside_node->name()]
            : inside_node->def();
    inside_node_def.set_input(inside_node_index, placeholder_name.c_str());

    VLOG(5) << "Input node def:" << inside_node_def.DebugString();
    VLOG(5) << " Place holder def :" << placeholder_def.DebugString();

    // Storing/updating  values in  nodedef map
    subgraph_nodes[inside_node->name()] = inside_node_def;
  }

  // Debug code
  VLOG(2) << " Incoming nodes in graphdef";
  for (auto it = subgraph_nodes.begin(); it != subgraph_nodes.end(); it++) {
    VLOG(5) << "name:" << it->first << "\n Nodedef:\n"
            << it->second.DebugString();
  }

  // Adding all the interior nodes to the list.
  for (auto node_id : *sg_params.subgraph_node_ids) {
    tensorflow::Node* node = sg_params.graph->FindNodeId(node_id);
    // this is to handle main graph output nodes in this segment.
    // filling map of (tensor_name , vector(dtype)) i.e A ->DT_FLOAT,->INT8
    // dtype order will match the the index.
    if (map_out_names_to_index.count(node->name())) {
      int num_outputs = map_out_names_to_index[node->name()];
      std::vector<PartialTensorShape> v_partial_shape;
      Status status =
          GetNodeAttr(node->attrs(), "_output_shapes", &v_partial_shape);
      if (!status.ok()) {
        status =
            GetNodeAttr(node->attrs(), kNeuronInferredShapes, &v_partial_shape);
      }
      if (!status.ok()) {
        v_partial_shape.resize(num_outputs);
      }
      for (auto port = 0; port <= num_outputs; ++port) {
        DataType tensor_dtype = node->output_type(port);
        main_graph_output_nodes[node->name()].push_back(
            std::make_pair(tensor_dtype, v_partial_shape[port]));
        VLOG(3) << " ConvertSubGraphToNeuronNodeDef node: making pair: ("
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

  VLOG(5) << "Neuron subgraph graphdef: " << subgraph_graph_def.DebugString();

  std::string in_graph_def_string = "";
  if (!subgraph_graph_def.SerializeToString(&in_graph_def_string)) {
    return tensorflow::errors::InvalidArgument("Failed to serialize subgraph");
  }

  // Gather output metadata
  VLOG(2) << "Neuron op: " << neuron_op_name
          << " sg_params.output_inds: " << sg_params.output_inds->size();
  std::vector<std::string> neuron_node_output_names;
  std::vector<tensorflow::DataType> neuron_node_output_dtypes;
  std::vector<PartialTensorShape> neuron_node_output_shapes;
  for (const std::pair<int, int>& output : *sg_params.output_inds) {
    int node_id = output.first;
    int output_idx = output.second;
    Node* node = sg_params.graph->FindNodeId(node_id);
    std::string op_name = node->name();
    std::string tensor_name = op_name + ":" + std::to_string(output_idx);

    VLOG(2) << "op_name: " << op_name;
    VLOG(2) << "Output tensor name: " << tensor_name;
    if (!main_graph_output_nodes.count(op_name)) {
      neuron_node_output_names.push_back(tensor_name);

      DataType tf_dtype = node->output_type(output_idx);
      neuron_node_output_dtypes.push_back(tf_dtype);

      std::vector<PartialTensorShape> v_partial_shape;
      if (GetNodeAttr(node->attrs(), "_output_shapes", &v_partial_shape).ok()) {
        neuron_node_output_shapes.push_back(v_partial_shape[output_idx]);
      } else if (GetNodeAttr(node->attrs(), kNeuronInferredShapes,
                             &v_partial_shape)
                     .ok()) {
        neuron_node_output_shapes.push_back(v_partial_shape[output_idx]);
      } else {
        neuron_node_output_shapes.emplace_back();
      }
    }
  }
  auto start_index = neuron_node_output_names.size();

  // Pushing the outputs of graph (if in the subgraph) to the output for the
  // subgraph. As we will create IdentityN connecting to the these nodes.
  for (auto name_type : main_graph_output_nodes) {
    uint counter = 0;
    auto name = name_type.first;
    for (auto dtype_shape : name_type.second) {
      auto dtype = dtype_shape.first;
      auto shape = dtype_shape.second;
      VLOG(3) << "tensorname: " << name << " dtype: " << dtype;
      std::string tensor_name = name + ":" + std::to_string(counter++);
      neuron_node_output_names.push_back(tensor_name);
      neuron_node_output_dtypes.push_back(dtype);
      neuron_node_output_shapes.push_back(shape);
    }
  }

  VLOG(2) << "Finished op preparation";

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

  VLOG(3) << "String to be hashed " << hash_string << "Hashed String "
          << hasher(hash_string) << "Hex Rep" << hex_string;

  neuron_op_name = "neuron_op_" + hex_string;
  VLOG(2) << "Hashed neuron_op_name: " << neuron_op_name;

  Node* neuron_node;
  TF_CHECK_OK(NodeBuilder(neuron_op_name, "NeuronOp")
                  .Input(input_nodes)
                  .Attr("graph_def", in_graph_def_string)
                  .Attr("input_names", input_names)
                  .Attr("input_dtypes", input_dtypes)
                  .Attr("input_shapes", input_shapes)
                  .Attr("output_names", neuron_node_output_names)
                  .Attr("output_dtypes", neuron_node_output_dtypes)
                  .Attr("output_shapes", neuron_node_output_shapes)
                  .Attr(kNeuronInferredShapes, neuron_node_output_shapes)
                  .Finalize(sg_params.graph, &neuron_node));

  sg_params.neuron_node = neuron_node;

  // Creating identityN node corresponding to any output nodes(if any) in this
  // subgraph
  // start_index is the index of Neuron op output to which IdentityN node should
  // be connected to.
  VLOG(3) << "start_index: " << start_index;
  std::vector<tensorflow::NodeBuilder::NodeOut> identity_inputs;

  // Iterate through the output node list found.
  for (auto name_index : main_graph_output_nodes) {
    identity_inputs.clear();
    auto name = name_index.first;
    VLOG(3) << " indentity inputs: name:" << name;
    VLOG(3) << " max index: " << map_out_names_to_index[name];
    for (size_t i = 0; i < main_graph_output_nodes[name].size(); ++i) {
      VLOG(3) << "start_index: " << start_index;
      identity_inputs.push_back(
          NodeBuilder::NodeOut(neuron_node, start_index++));
    }
    Node* node;
    TF_CHECK_OK(NodeBuilder(name, "IdentityN")
                    .Input(identity_inputs)
                    .Finalize(sg_params.graph, &node));
    VLOG(3) << " New output IdentityN node: " << node->def().DebugString();
  }

  VLOG(3) << "Created new node ..............";
  VLOG(5) << " new node: " << neuron_node->def().DebugString();

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
static tensorflow::Status FillSubGraphEdgeSets(ConvertGraphParams* params) {
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
tensorflow::Status ConvertSubGraphToNeuron(ConvertGraphParams* params) {
  TF_RETURN_IF_ERROR(FillSubGraphEdgeSets(params));
  tensorflow::NodeDef neuron_node_def;
  tensorflow::Node* neuron_node = nullptr;

  SubGraphParams sg_params(*params->graph, *params->subgraph_node_ids,
                           *params->output_names, params->subgraph_outputs,
                           params->subgraph_incoming_edges, neuron_node,
                           params->neuron_op_count);

  TF_RETURN_IF_ERROR(ConvertSubGraphToNeuronNodeDef(sg_params));

  neuron_node_def = sg_params.neuron_node->def();
  neuron_node = sg_params.neuron_node;

  // AddNode does not wire edges.
  // Re-map incoming edges to use the new Neuron node instead of the orig
  // subgraph
  for (const tensorflow::Edge* edge : params->subgraph_incoming_edges) {
    if (edge->IsControlEdge()) {
      params->graph->AddControlEdge(edge->src(), neuron_node);
    }
    params->graph->RemoveEdge(edge);
  }

  VLOG(3) << "new wiring edges: " << neuron_node->in_edges().size();
  for (const tensorflow::Edge* edge : neuron_node->in_edges()) {
    VLOG(3) << edge->src()->name() << " port: " << edge->src_output();
  }

  // Re-map outgoing edges to use the new Neuron node instead of the orig
  // subgraph
  for (const tensorflow::Edge* edge : params->subgraph_outgoing_edges) {
    if (edge->IsControlEdge()) {
      params->graph->AddControlEdge(neuron_node, edge->dst());
    } else {
      std::string old_src_name =
          edge->src()->name() + ":" + std::to_string(edge->src_output());
      VLOG(3) << "Old src: " << old_src_name;
      auto& output_names = neuron_node_def.attr().at("output_names").list().s();
      auto iter =
          std::find(output_names.begin(), output_names.end(), old_src_name);
      int new_src_output = 0;
      if (iter == output_names.end()) {
        return errors::Internal("Old src name ", old_src_name,
                                " not found among outputs of ",
                                neuron_node->name());
      } else {
        new_src_output = std::distance(output_names.begin(), iter);
      }
      VLOG(3) << "Updating " << neuron_node->name() << ":" << new_src_output
              << " --> " << edge->dst()->name();
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
    TF_RETURN_IF_ERROR(ExcludeInputNodes(in_node, exclude_node_list));
  }

  return tensorflow::Status::OK();
}

// Takes all the segment and modifies input graph to have Neuron ops.
static tensorflow::Status ProcessSegments(
    tensorflow::Graph& graph, const std::vector<string>& output_names,
    std::unordered_map<string, tensorflow::Node*>& node_map,
    tensorflow::tensorrt::segment::SegmentNodesVector& segments) {
  tensorflow::Status status = tensorflow::Status::OK();
  int neuron_op_index = 0;

  for (const std::set<const Node*>& subgraph_node_names : segments) {
    std::set<int> subgraph_node_ids;
    std::stringstream oss;
    for (const Node* node : subgraph_node_names) {
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
    VLOG(2) << "Subgraph num nodes" << subgraph_node_ids.size();
    VLOG(3) << "Subgraph nodes" << oss.str();

    ConvertGraphParams params(graph, output_names, subgraph_node_ids,
                              neuron_op_index++);
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
    tensorflow::Graph& graph,
    tensorflow::tensorrt::segment::SegmentNodesVector& normal_segments) {
  /*
  For each segment:
    1. If a resouce is input or output of segment, add segment to remove list.
    2. Replace any Neuron ops that is inside the segment with actual nodes. This
  is happens coz we we are currently doing segmentation on graph that might have
    loops inside Neuron ops.
    3. Remove all segments in remove list from segment list.
  */
  std::unordered_map<string, tensorflow::Node*> node_map;
  TF_LOG_IF_ERROR(BuildNodeMap(graph, &node_map));
  std::set<int> remove_segment_index;
  for (auto idx = 0; idx < (int)normal_segments.size(); idx++) {
    std::set<const Node*>& nodes_in_segment = normal_segments[idx];
    std::set<int> subgraph_node_ids;
    for (auto node : nodes_in_segment) {
      subgraph_node_ids.insert(node_map[node->name()]->id());
    }

    tensorflow::EdgeSet incoming_edges;
    tensorflow::EdgeSet outgoing_edges;
    GetSubGraphIncomingEdges(graph, subgraph_node_ids, &incoming_edges);
    GetSubGraphOutgoingEdges(graph, subgraph_node_ids, &outgoing_edges);

    for (const tensorflow::Edge* edge : incoming_edges) {
      if (edge->dst()->input_type(edge->dst_input()) == DT_RESOURCE) {
        VLOG(3) << "incoming edge src: " << edge->src()->name()
                << " dst: " << edge->dst()->name();
        remove_segment_index.insert(idx);
        break;
      }
    }

    for (const tensorflow::Edge* edge : outgoing_edges) {
      if (edge->src()->output_type(edge->src_output()) == DT_RESOURCE) {
        VLOG(3) << "outgoing edge src: " << edge->src()->name()
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
    VLOG(3) << " Removing segment :" << idx
            << " num of segment nodes : " << normal_segments[rel_idx].size();
    normal_segments.erase(normal_segments.begin() + rel_idx);
  }
}

// Performing :
// 1. Clearing device info from all nodes
// 2. Removing _class:loc attr from nodedef
Status PreProcessingGraphDef(GraphDef& in_graph_def) {
  VLOG(1) << " Creating Identities for all outputs";
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

  VLOG(5) << in_graph_def.DebugString();
  return tensorflow::Status::OK();
}

static Status FindConstantFoldableNodes(
    std::unordered_set<std::string>* foldable_nodes,
    const GraphDef& graph_def) {
  // TODO: determine if we need grappler::TopologicalSort
  std::unordered_map<std::string, const NodeDef*> name_to_node;
  for (const auto& node : graph_def.node()) {
    VLOG(3) << "adding node " << node.name();
    name_to_node[node.name()] = &node;
  }
  for (const auto& node : graph_def.node()) {
    bool foldable = false;
    if (node.op() == "Shape" || node.op() == "Size") {
      VLOG(3) << "looking at input " << node.input(0);
      auto in_name_port = ParseTensorName(node.input(0));
      std::string in_name = in_name_port.first;
      int in_port = in_name_port.second;
      const NodeDef* in_node = name_to_node.at(in_name);
      const auto& attr = in_node->attr();
      const auto& shape = attr.at(kNeuronInferredShapes).list().shape(in_port);
      foldable = PartialTensorShape(shape).IsFullyDefined();
      VLOG(3) << "node " << node.name() << ", foldable " << foldable;
    } else {
      const auto& inputs = node.input();
      auto predicate = [foldable_nodes,
                        &name_to_node](const std::string& input_name) {
        std::string node_name(input_name);
        size_t control_start = node_name.find('^');
        if (control_start != std::string::npos) {
          node_name = node_name.substr(control_start + 1);
        }
        size_t index_start = node_name.find(':');
        if (index_start != std::string::npos) {
          node_name = node_name.substr(0, index_start);
        }
        VLOG(3) << "determining status of node " << node_name;
        return foldable_nodes->count(node_name) ||
               name_to_node.at(node_name)->op() == "Const";
      };
      foldable = node.input_size() &&
                 std::all_of(inputs.begin(), inputs.end(), predicate);
    }
    if (foldable) {
      VLOG(3) << "found constant-foldable node " << node.name();
      foldable_nodes->insert(node.name());
    }
  }
  return Status::OK();
}

namespace {

void PruneCheapSegments(tensorrt::segment::SegmentNodesVector& segments,
                        const std::set<std::string>& expensive_op_types) {
  int total_expensive_op_count = 0;
  for (std::set<const Node*>& nodes : segments) {
    for (const Node* node : nodes) {
      total_expensive_op_count += expensive_op_types.count(node->type_string());
    }
  }
  if (0 == total_expensive_op_count) {
    VLOG(1) << "No expensive operator found in the entire graph -- do nothing";
    return;
  }
  for (std::set<const Node*>& nodes : segments) {
    int expensive_op_count = 0;
    for (const Node* node : nodes) {
      expensive_op_count += expensive_op_types.count(node->type_string());
    }
    if (0 == expensive_op_count) {
      nodes.clear();
    }
  }
  tensorrt::segment::SegmentNodesVector new_segments;
  for (std::set<const Node*>& nodes : segments) {
    if (!nodes.empty()) {
      new_segments.push_back(nodes);
    }
  }
  segments = new_segments;
}

}  // namespace

// This function is the base function which does:
// Step 1: Find Neuron Segments.
// Step 2: Calls functions to create Neuron subgraphs.
Status CreateNeuronGraphDef(GraphDef* new_graph_def, const GraphDef& graph_def,
                            const std::vector<std::string>& input_op_names,
                            const std::vector<std::string>& output_op_names,
                            const bool fuse_foldable_nodes,
                            const int minimum_segment_size,
                            const double prune_small_subgraphs_ratio,
                            const std::set<std::string>& supported_op_types,
                            const std::set<std::string>& no_fuse_ops,
                            const std::set<std::string>& force_fuse_ops,
                            const std::set<std::string>& expensive_op_types) {
  // Segment the graph into subgraphs that can be converted to Neuron op
  tensorflow::tensorrt::segment::SegmentOptions segment_options;

  VLOG(1) << "Building Neuron Op\n";

  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             graph_def.library());

  GraphDef temp_graph_def;
  temp_graph_def.CopyFrom(graph_def);
  TF_RETURN_IF_ERROR(PreProcessingGraphDef(temp_graph_def));

  tensorflow::Graph graph(flib);

  TF_CHECK_OK(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), temp_graph_def, &graph));

  // Build output tensor names
  std::unordered_map<std::string, const Node*> op_name_to_node;
  for (const Node* node : graph.op_nodes()) {
    op_name_to_node[node->name()] = node;
  }
  std::vector<std::string> outputs;
  for (const auto& op_name : output_op_names) {
    const Node* node = op_name_to_node[op_name];
    int64 num_outputs = node->num_outputs();
    VLOG(3) << "Output " << op_name << " contains " << num_outputs
            << " outputs";
    for (int64 idx = 0; idx < num_outputs; ++idx) {
      outputs.push_back(op_name + ":" + std::to_string(idx));
    }
  }

  // Find "constant-foldable" nodes and claim them as supported
  std::unordered_set<std::string> foldable_nodes;
  if (fuse_foldable_nodes) {
    TF_RETURN_IF_ERROR(FindConstantFoldableNodes(&foldable_nodes, graph_def));
  }

  std::unordered_map<std::string, tensorflow::Node*> node_map;
  TF_RETURN_IF_ERROR(BuildNodeMap(graph, &node_map));

  segment_options.minimum_segment_size = minimum_segment_size;

  // Setup exclude_node_list
  for (Node* node : graph.nodes()) {
    bool is_source_or_sink = node->IsSink() || node->IsSource();
    bool is_supported = supported_op_types.count(node->type_string());
    bool no_fuse = no_fuse_ops.count(node->name());
    bool force_fuse = force_fuse_ops.count(node->name());
    bool is_foldable = foldable_nodes.count(node->name());
    bool supported_can_fuse = is_supported && !is_source_or_sink && !no_fuse;
    bool fuseable = supported_can_fuse || force_fuse || is_foldable;
    if (node->def().attr().count(kNeuronInFixedShapeContext)) {
      bool fixed_shape = node->def().attr().at(kNeuronInFixedShapeContext).b();
      VLOG(2) << "Node " << node->name() << " fixed_shape=" << fixed_shape;
      fuseable &= fixed_shape;
    }
    if (!fuseable) {
      VLOG(2) << "Node " << node->name() << " will not be fused";
      segment_options.exclude_node_list.insert(node->name());
    } else {
      VLOG(2) << "Will try fusing node " << node->name();
    }
  }

  // All inout nodes to exclude list
  for (auto node_name : input_op_names) {
    segment_options.exclude_node_list.insert(node_name);

    // Adding all the nodes before the input node to exclude list.
    tensorflow::Node* omit_node = node_map[node_name];
    if (omit_node) {
      TF_RETURN_IF_ERROR(
          ExcludeInputNodes(omit_node, segment_options.exclude_node_list));
    }
  }

  tensorflow::tensorrt::segment::SegmentNodesVector segments;
  std::function<bool(const Edge*)> input_edge_validator;
  std::function<bool(const Edge*)> output_edge_validator;
  if (force_fuse_ops.size()) {
    // Don't exclude edges if manual segmentation is specified
    input_edge_validator = [](const Edge* edge) { return true; };
    output_edge_validator = [](const Edge* edge) { return true; };
  } else {
    input_edge_validator = EdgeValidator();
    output_edge_validator = OutputEdgeValidator();
  }

  TF_RETURN_IF_ERROR(tensorflow::tensorrt::segment::SegmentGraph(
      &graph, [](const Node* node) { return Status::OK(); },
      input_edge_validator, output_edge_validator, segment_options, &segments));
  if (segments.size() > 1 && !expensive_op_types.empty()) {
    VLOG(1) << "MULTIPLE Neuron candidates -- pruning away those without"
            << " any arithmetic-intensive operator";
    PruneCheapSegments(segments, expensive_op_types);
  }
  if (segments.size() > 1) {
    VLOG(1) << "MULTIPLE Neuron candidate conversion: " << segments.size();
    if (prune_small_subgraphs_ratio < 0.0 ||
        prune_small_subgraphs_ratio > 1.0) {
      return errors::Internal("Found invalid prune_small_subgraphs_ratio ",
                              prune_small_subgraphs_ratio);
    }
    if (prune_small_subgraphs_ratio > 0.0) {
      size_t size_all_segments = 0;
      for (const auto& seg : segments) {
        size_all_segments += seg.size();
      }
      VLOG(1) << "Total size of all segments: " << size_all_segments;
      auto comp = [](const std::set<const Node*>& lhs,
                     const std::set<const Node*>& rhs) {
        return lhs.size() < rhs.size();
      };
      auto max_segment =
          *std::max_element(segments.begin(), segments.end(), comp);
      VLOG(1) << "Maximum segment size " << max_segment.size();
      if (((double)max_segment.size() / (double)size_all_segments) >
          prune_small_subgraphs_ratio) {
        VLOG(1) << "Only keep maximum segment with size " << max_segment.size();
        segments.clear();
        segments.push_back(max_segment);
      }
    }
  }

  if (segments.size()) {
    PreProcessSegmentsForResources(graph, segments);
    TF_RETURN_IF_ERROR(ProcessSegments(graph, outputs, node_map, segments));
  }

  graph.ToGraphDef(new_graph_def);
  VLOG(5) << "new_graph_def: " << new_graph_def->DebugString();
  return tensorflow::Status::OK();
}

}  // namespace convert
}  // namespace neuron
}  // namespace tensorflow

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
#include "tensorflow/python/neuron/convert/convert_graph.h"
#include "tensorflow/python/neuron/segment/segment.h"
#include "tensorflow/python/neuron/util/logging.h"

namespace tensorflow {
namespace kaena {
namespace convert {

using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

bool IsEIACandidate(const tensorflow::Node* node,
                    const std::set<std::string>& op_whitelist,
                    const std::set<std::string>& op_cpu,
                    const std::set<std::string>& op_inferentia) {
  bool isSourceorSink = node->IsSink() || node->IsSource();
  bool isWhitelisted = op_whitelist.count(node->type_string());
  bool isOnCPU = op_cpu.count(node->name());
  bool isOnInferentia = op_inferentia.count(node->name());
  return (isWhitelisted && !isSourceorSink && !isOnCPU) || isOnInferentia;
}

// Helper function to split tensor and tensor spec (i.e conv:2 -> conv, 2)
std::vector<string> split(string str, string token) {
  std::vector<string> result;
  while (str.size()) {
    int index = str.find(token);
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
static string EdgeIter(const Graph& g) {
  std::vector<std::pair<int, int>> edges;
  for (const Edge* e : g.edges()) {
    edges.push_back(std::make_pair(e->src()->id(), e->dst()->id()));
  }
  std::sort(edges.begin(), edges.end());
  string result;
  for (auto& p : edges) {
    strings::StrAppend(&result, p.first, "->", p.second, ";");
  }
  return result;
}

// Helper function
Status MakePlaceholder(string name, Node* old_node, Node** new_node, Graph* g,
                       DataType n_type, int output_index = 0)
/*
 * Function to make a placeholder with specific attributes, Currently only
 * copies shape
 * Input:
 *  Name: of the place holder
 *  old_node: Original Node
 *  Graph g: graph to create nodes in
 *  n_type: Type of the placeholder
 *  output_index: which output we are creating the placeholder for.. Shouldn't
 * need this if no vector
 * Output:
 *  new_node: New node that will be created
 *
 *  Return:
 *    Status of Creation
 *
 */
{
  AttrSlice attrs = old_node->attrs();
  TensorShape tensor_shape;
  PartialTensorShape partial_shape;
  std::vector<PartialTensorShape> v_partial_shape;
  std::vector<TensorShape> v_tensor_shape;
  Status s;
  if ((GetNodeAttr(attrs, "shape", &tensor_shape).ok())) {
    s = NodeBuilder(name, "Placeholder")
            .Attr("shape", tensor_shape)
            .Attr("dtype", n_type)
            .Finalize(g, new_node);
  } else if ((GetNodeAttr(attrs, "shape", &partial_shape).ok())) {
    s = NodeBuilder(name, "Placeholder")
            .Attr("shape", partial_shape)
            .Attr("dtype", n_type)
            .Finalize(g, new_node);
  } else if ((GetNodeAttr(attrs, "output_shapes", &v_tensor_shape).ok())) {
    s = NodeBuilder(name, "Placeholder")
            .Attr("shape", v_tensor_shape[output_index])
            .Attr("dtype", n_type)
            .Finalize(g, new_node);
  } else if ((GetNodeAttr(attrs, "output_shapes", &v_partial_shape).ok())) {
    EILOG(1) << v_partial_shape[output_index];
    s = NodeBuilder(name, "Placeholder")
            .Attr("shape", v_partial_shape[output_index])
            .Attr("dtype", n_type)
            .Finalize(g, new_node);
  } else {
    s = NodeBuilder(name, "Placeholder")
            .Attr("dtype", n_type)
            .Finalize(g, new_node);
  }
  EILOG(1) << new_node ? (*new_node)->DebugString()
                       : "MakePlaceholder failed. new_node is NULL";
  TF_CHECK_OK(s);
  return s;
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
        EILOG(2) << "src: " << edge->src()->name()
                 << " dst: " << edge->dst()->name();
        incoming_edges->insert(edge);
      } else {
        EILOG(2) << node->name() << " -> " << edge->src()->name() << " N, ";
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
        EILOG(2) << node->name() << " -> " << edge->dst()->name() << " Y, ";
        outgoing_edges->insert(edge);
      } else {
        EILOG(2) << node->name() << " -> " << edge->dst()->name() << " N, ";
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

static void printDataTypeVector(std::vector<tensorflow::DataType>& dt_vec) {
  EILOG(2) << "printing data type vector" << std::endl;
  for (tensorflow::DataType dt : dt_vec) {
    EILOG(2) << dt << std::endl;
  }
}

static void printNodeOutVector(std::vector<NodeBuilder::NodeOut>& node_vec) {
  for (NodeBuilder::NodeOut node : node_vec) {
    EILOG(2) << node.name << std::endl;
  }
}

static void printNameVector(std::vector<string>& name_vec) {
  EILOG(2) << "printing name vector" << std::endl;
  for (string name : name_vec) {
    EILOG(2) << name << std::endl;
  }
}

static void printNodes(const Graph& graph) {
  EILOG(2) << "printing nodes" << std::endl;
  for (Node* node : graph.nodes()) {
    if (node) EILOG(2) << node->name() << std::endl;
  }
}

static void printGraphDefNodes(const GraphDef& gdef) {
  for (int i = 0; i < gdef.node_size(); i++) {
    EILOG(2) << "node: " << gdef.node(i).name();
  }
}

// helper function
template <class T>
bool IsPresentinMap(const T& map_list, const string& value) {
  return (map_list.find(value) != map_list.end());
}

static void GetOutputInforamtion(const tensorflow::Graph& graph,
                                 const std::set<int>& subgraph_node_ids,
                                 std::vector<string>& output_names,
                                 std::vector<tensorflow::DataType>& outT) {
  for (unsigned int ii = 0; ii < output_names.size(); ii++) {
    EILOG(2) << subgraph_node_ids.size() << " Looking for " << output_names[ii]
             << "\n";
    for (int node_id : subgraph_node_ids) {
      tensorflow::Node* node = graph.FindNodeId(node_id);
      string name = output_names[ii];
      EILOG(2) << "curr node name:  " << node->name() << "\n";
      if (name.compare(node->name()) == 0) {
        Node* tmp;
        // Add Node to input and copy types and things
        outT.push_back(node->output_type(0));
      }
    }
  }
}

string GetCommonNameScope(const string& op_name_a, const string& op_name_b) {
  size_t last_scope_separator = 0;
  for (size_t i = 0; i < std::min(op_name_a.size(), op_name_b.size()); ++i) {
    if (op_name_a[i] != op_name_b[i]) {
      break;
    } else if (op_name_a[i] == '/') {
      last_scope_separator = i + 1;
    }
  }
  return op_name_a.substr(0, last_scope_separator);
}

static void printGraphEdges(Graph* graph) {
  for (tensorflow::Edge* edge : graph->edges()) {
    EILOG(2) << "edge " << edge->id() << ": " << edge->src()->name() << ":"
             << edge->src_output() << " (" << edge->src()->op_def().name()
             << ") -> " << edge->dst()->name() << ":" << edge->dst_input()
             << " (" << edge->dst()->op_def().name() << ") "
             << "Control Edge: " << edge->IsControlEdge();
  }
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
tensorflow::Status ConvertSubGraphToEIANodeDef(SubGraphParams& s) {
  string eop_name = StrCat("inferentia_op_", s.eop_index);
  EILOG(1) << "Start Node building ...." << eop_name;

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

  for (auto out_tensor : s.output_names) {
    auto tensor_vec = split_tensor(out_tensor);
    map_out_names_to_index[tensor_vec[0]] = std::max(
        map_out_names_to_index[tensor_vec[0]], std::stoi(tensor_vec[1]));
  }

  // Adding placeholder for all incoming edges, since
  // placeholders do not need to specify inputs.
  for (auto incoming_edge : s.subgraph_incoming_edges) {
    if (incoming_edge->IsControlEdge()) {
      EILOG(2) << "Control edge: src-> " << incoming_edge->src()->name()
               << " dst-> " << incoming_edge->dst()->name();

      // Need to remove this input from the node. Control edges incoming to
      // the subgraph should point to eia op directly. The inputs in nodes
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

    tensorflow::Node* inside_node = s.graph.FindNodeId(inside_node_id);
    tensorflow::Node* outside_node = s.graph.FindNodeId(outside_node_id);

    EILOG(2) << "edge->  src:" << outside_node->name()
             << " idx: " << outside_node_index << " dst:" << inside_node->name()
             << " idx: " << inside_node_index;

    NodeDef placeholder_def;
    auto placeholder_name = s.graph.NewName(
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

    EILOG(2) << "Input node def:" << inside_node_def.DebugString();
    EILOG(2) << " Place holder def :" << placeholder_def.DebugString();

    // for creating EIA op node,storing:
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
  EILOG(2) << " Incoming nodes in graphdef";
  for (auto it = subgraph_nodes.begin(); it != subgraph_nodes.end(); it++) {
    EILOG(2) << "name:" << it->first << "\n Nodedef:\n"
             << it->second.DebugString();
  }

  // Adding all the interior nodes to the list.
  for (auto node_id : s.subgraph_node_ids) {
    tensorflow::Node* node = s.graph.FindNodeId(node_id);
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
        EILOG(1) << " ConvertSubGraphToEIANodeDef node: making pair: ("
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

  EILOG(2) << "EI subgraph graphdef: " << subgraph_graph_def.DebugString();
  printGraphDefNodes(subgraph_graph_def);

  string in_graph_def_string = "";
  if (!subgraph_graph_def.SerializeToString(&in_graph_def_string)) {
    return tensorflow::errors::InvalidArgument("Failed to serialize subgraph");
  }

  // Gather output metadata
  EILOG(2) << "eia op: " << eop_name
           << " s.output_inds: " << s.output_inds.size();
  std::vector<string> ei_node_output_names;
  std::vector<tensorflow::DataType> ei_node_output_dtypes;
  std::vector<PartialTensorShape> ei_node_output_shapes;
  for (const std::pair<int, int>& output : s.output_inds) {
    int node_id = output.first;
    int output_idx = output.second;
    tensorflow::Node* node = s.graph.FindNodeId(node_id);
    string op_name = node->name();
    string tensor_name = op_name;

    EILOG(2) << "op_name: " << op_name;
    tensorflow::strings::StrAppend(&tensor_name, ":", output_idx);
    EILOG(2) << "Output tensor name: " << tensor_name;
    ei_node_output_names.push_back(tensor_name);

    tensorflow::DataType tf_dtype = node->output_type(output_idx);
    ei_node_output_dtypes.push_back(tf_dtype);

    std::vector<PartialTensorShape> v_partial_shape;
    if (GetNodeAttr(node->attrs(), "_output_shapes", &v_partial_shape).ok()) {
      ei_node_output_shapes.push_back(v_partial_shape[output_idx]);
    } else {
      ei_node_output_shapes.push_back(PartialTensorShape());
    }
  }

  // Pushing the outputs of graph (if in the subgraph) to the output for the
  // subgraph. As we will create IdentityN connecting to the these nodes.
  for (auto name_type : main_graph_output_nodes) {
    uint counter = 0;
    auto name = name_type.first;
    auto dtype = name_type.second;
    for (auto dtype : name_type.second) {
      EILOG(0) << "tensorname: " << name << " dtype: " << dtype;
      ei_node_output_names.push_back(name + ":" + std::to_string(counter++));
      ei_node_output_dtypes.push_back(dtype);
    }
    for (auto shape : main_graph_output_shapes[name]) {
      ei_node_output_shapes.push_back(shape);
    }
  }

  EILOG(1) << "Finished op preparation";

  StringPieceHasher hasher;
  std::string hash_string = "";
  for (auto const& s : input_names) {
    hash_string += s;
  }
  for (auto const& s : ei_node_output_names) {
    hash_string += s;
  }
  for (auto const& s : ei_node_output_dtypes) {
    hash_string += s;
  }
  std::stringstream stream;
  stream << std::hex << hasher(hash_string);
  std::string hex_string(stream.str());

  EILOG(2) << "String to be hashed " << hash_string << "Hashed String "
           << hasher(hash_string) << "Hex Rep" << hex_string;

  eop_name = "inferentia_op_" + hex_string;

  s.eop_index_to_name_map->insert({hex_string, s.eop_index});

  // Debug code to dump all subgraphs.
  // string out_graph =
  //     "/home/ubuntu/subgraphs/" + std::to_string(s.eop_index) + ".pb";
  // WriteBinaryProto(Env::Default(), out_graph, subgraph_graph_def);

  // todo: need ei_node_output_shapes to construct output tensors until krtd has shapes
  Node* eia_node;
  TF_CHECK_OK(NodeBuilder(eop_name, "InferentiaOp")
                  .Input(input_nodes)
                  .Attr("graph_def", in_graph_def_string)
                  .Attr("input_names", input_names)
                  .Attr("input_dtypes", input_dtypes)
                  .Attr("input_shapes", input_shapes)
                  .Attr("output_names", ei_node_output_names)
                  .Attr("output_dtypes", ei_node_output_dtypes)
                  .Attr("output_shapes", ei_node_output_shapes)
                  .Attr("_output_shapes", ei_node_output_shapes)
                  .Finalize(&s.graph, &eia_node));

  s.eia_node = eia_node;

  // Creating identityN node corresponding to any output nodes(if any) in this
  // subgraph
  // start_index is the index of ei op output to which IdentityN node should be
  // connected to.
  auto start_index = s.output_inds.size();
  EILOG(1) << "start_index: " << start_index;
  std::vector<tensorflow::NodeBuilder::NodeOut> identity_inputs;

  // Iterate through the output node list found.
  for (auto name_index : main_graph_output_nodes) {
    identity_inputs.clear();
    auto name = name_index.first;
    EILOG(1) << " indentity inputs: name:" << name;
    EILOG(1) << " max index: " << map_out_names_to_index[name];
    for (int i = 0; i < main_graph_output_nodes[name].size(); ++i) {
      EILOG(1) << "start_index: " << start_index;
      identity_inputs.push_back(NodeBuilder::NodeOut(eia_node, start_index++));
    }
    Node* node;
    TF_CHECK_OK(NodeBuilder(name, "IdentityN")
                    .Input(identity_inputs)
                    .Finalize(&s.graph, &node));
    EILOG(1) << " New output IdentityN node: " << node->def().DebugString();
  }

  EILOG(1) << "Created new node ..............";
  EILOG(2) << " new node: " << eia_node->def().DebugString();

  return tensorflow::Status::OK();
}

std::unordered_map<string, std::vector<int>> BuildTensorNameMap(
    const std::vector<string>& tensor_names) {
  std::unordered_map<string, std::vector<int>> result;
  for (string const& tensor_name : tensor_names) {
    string node_name;
    int index;
    std::tie(node_name, index) = ParseTensorName(tensor_name);
    EILOG(2) << "node_name : " << node_name << " index:" << index;
    result[node_name].push_back(index);
  }
  return result;
}

// This fills the ConvertGraphParams struct.
static tensorflow::Status FillSubGraphEdgeSets(ConvertGraphParams* p) {
  GetSubGraphIncomingEdges(p->graph, p->subgraph_node_ids,
                           &p->subgraph_incoming_edges);

  auto output_name_to_index_map = BuildTensorNameMap(p->output_names);
  std::set<std::pair<int, int>> subgraph_outputs_set;
  // Collect outputs referenced from output_names
  GetSubGraphOutgoingEdges(p->graph, p->subgraph_node_ids,
                           &p->subgraph_outgoing_edges);
  for (const tensorflow::Edge* edge : p->subgraph_outgoing_edges) {
    if (!edge->IsControlEdge())
      subgraph_outputs_set.insert({edge->src()->id(), edge->src_output()});
  }
  p->subgraph_outputs.reserve(subgraph_outputs_set.size());
  p->subgraph_outputs.insert(p->subgraph_outputs.begin(),
                             subgraph_outputs_set.begin(),
                             subgraph_outputs_set.end());
  return tensorflow::Status::OK();
}

// Sets up structs for creating subgraph.
// Also, rewires new eia node to the graph. Removes old nodes.
tensorflow::Status ConvertSubGraphToEIA(ConvertGraphParams* params) {
  TF_RETURN_IF_ERROR(FillSubGraphEdgeSets(params));
  tensorflow::NodeDef eia_node_def;
  tensorflow::Node* eia_node;

  SubGraphParams s(params->graph, params->subgraph_node_ids,
                   params->output_names, params->subgraph_outputs,
                   params->subgraph_incoming_edges, eia_node, params->eop_count,
                   params->eop_index_to_name_map, params->p_model_id,
                   params->precision_mode);

  TF_RETURN_IF_ERROR(ConvertSubGraphToEIANodeDef(s));

  tensorflow::Status status;
  eia_node_def = s.eia_node->def();
  eia_node = s.eia_node;

  // AddNode does not wire edges.
  // Re-map incoming edges to use the new eia node instead of the orig subgraph
  std::map<std::pair<int, int>, int> subgraph_edge_to_input_map;
  EILOG(2) << "subgraph_edge_to_input_map: ";
  for (size_t i = 0; i < params->subgraph_inputs.size(); ++i) {
    EILOG(2) << params->subgraph_inputs.at(i).first << " , "
             << params->subgraph_inputs.at(i).second << " i " << i;
    subgraph_edge_to_input_map.insert({params->subgraph_inputs.at(i), i});
  }
  for (const tensorflow::Edge* edge : params->subgraph_incoming_edges) {
    // std::pair<int, int> old_src = {edge->src()->id(), edge->src_output()};
    // int new_src_output = subgraph_edge_to_input_map.at(old_src);
    if (edge->IsControlEdge()) {
      params->graph.AddControlEdge(edge->src(), eia_node);
    }
    params->graph.RemoveEdge(edge);
  }

  EILOG(2) << "new wiring edges: " << eia_node->in_edges().size();
  for (const tensorflow::Edge* edge : eia_node->in_edges()) {
    EILOG(2) << edge->src()->name() << " port: " << edge->src_output();
  }

  TF_RETURN_IF_ERROR(status);

  // Re-map outgoing edges to use the new eia node instead of the orig subgraph
  std::map<std::pair<int, int>, int> subgraph_edge_to_output_map;
  for (size_t i = 0; i < params->subgraph_outputs.size(); ++i) {
    subgraph_edge_to_output_map.insert({params->subgraph_outputs.at(i), i});
  }
  TF_RETURN_IF_ERROR(status);
  for (const tensorflow::Edge* edge : params->subgraph_outgoing_edges) {
    if (edge->IsControlEdge()) {
      params->graph.AddControlEdge(eia_node, edge->dst());
    } else {
      std::pair<int, int> old_src = {edge->src()->id(), edge->src_output()};
      int new_src_output = subgraph_edge_to_output_map.at(old_src);
      TF_RETURN_IF_ERROR(params->graph.UpdateEdge(
          eia_node, new_src_output, edge->dst(), edge->dst_input()));
    }
  }

  // Remove the original subgraph
  for (int node_id : params->subgraph_node_ids) {
    tensorflow::Node* node = params->graph.FindNodeId(node_id);
    // Don't remove the input placeholders
    if (node->type_string() == "Placeholder") {
      continue;
    }
    params->graph.RemoveNode(node);
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

// Takes all the segment and modifies input graph to have eia ops.
static tensorflow::Status ProcessSegments(
    tensorflow::Graph& graph, const std::vector<string>& output_names,
    std::unordered_map<string, tensorflow::Node*>& node_map,
    tensorflow::kaena::segment::SegmentNodesVector& segments, int& eop_index,
    std::unordered_map<string, int>* eop_index_to_name_map, string model_id) {
  tensorflow::Status status = tensorflow::Status::OK();

  for (const std::set<string>& subgraph_node_names : segments) {
    std::set<int> subgraph_node_ids;
    std::stringstream oss;
    for (const string& node_name : subgraph_node_names) {
      oss << " " << node_name;
      if (!IsPresentinMap(node_map, node_name)) {
        string msg =
            "Failed to find node in the graph while creating "
            "processing segments !!!!";
        EILOG(0) << msg;
        return tensorflow::Status(tensorflow::error::Code::INTERNAL, msg);
      }
      subgraph_node_ids.insert(node_map.at(node_name)->id());
    }
    EILOG(1) << "Subgraph num nodes" << subgraph_node_ids.size();
    EILOG(2) << "Subgraph nodes" << oss.str();
    // defaulting to FP32 , this var is not used right now.
    int precision_mode = FP32MODE;

    ConvertGraphParams p(graph, output_names, subgraph_node_ids, precision_mode,
                         eop_index++, eop_index_to_name_map, model_id);

    if (precision_mode == FP16MODE) {
      // TODO: Can be used to convert to FP16 possibly later.
    } else {
      tensorflow::Status status = ConvertSubGraphToEIA(&p);
      if (status != tensorflow::Status::OK()) {
        LOG(WARNING) << "subgraph conversion error for subgraph_index:"
                     << eop_index - 1 << " due to: \"" << status.ToString()
                     << "\" SKIPPING......( " << subgraph_node_names.size()
                     << " nodes)";
      }
    }
  }

  return status;
}

void ReplaceLoopEIAOpsWithNodes(
    std::set<std::string>& segment_nodes,
    tensorflow::kaena::segment::SegmentNodesVector& loop_segments,
    std::unordered_map<string, int>* eop_index_to_name_map) {
  // Parse through all the segment nodes and if it contains op names "eop_x",
  // replace that with ops in loop_segments index 'x' ops.
  std::set<std::string> internal_eop_ops;
  std::set<int> eop_index_to_be_added;
  for (auto node_name : segment_nodes) {
    if (node_name.find("eop") != std::string::npos) {
      internal_eop_ops.insert(node_name);
      EILOG(2) << " eop op " << node_name;
      // loop_segments index will match the eop name index i.e 'x' in 'eop_x'
      // since we created eop_x while iterating through loop_segments
      std::size_t found = node_name.find_last_of("_");
      // std::string parent_name = node_names.substr(0, found);
      auto loop_segment_index = (*eop_index_to_name_map)[node_name.substr(
          found + 1, node_name.size())];
      eop_index_to_be_added.insert(loop_segment_index);
      EILOG(0) << " loop_segment index: " << loop_segment_index;
    }
  }

  // removing eia ops from current segment.
  for (auto name : internal_eop_ops) {
    segment_nodes.erase(name);
  }

  // Adding all the nodes for corresponding eia ops to current segment.
  for (auto eop_idx : eop_index_to_be_added) {
    segment_nodes.insert(loop_segments[eop_idx].begin(),
                         loop_segments[eop_idx].end());
  }

  // // clearing storage:
  // internal_eop_ops.clear();
  // eop_index_to_be_added.clear();
}

void PreProcessSegmentsForResources(
    tensorflow::Graph& graph,
    tensorflow::kaena::segment::SegmentNodesVector& normal_segments,
    tensorflow::kaena::segment::SegmentNodesVector& loop_segments,
    // uint& ops_kept_local,
    std::unordered_map<string, int>* eop_index_to_name_map) {
  /*
  For each segment:
    1. If a resouce is input or output of segment, add segment to remove list.
    2. Replace any eia ops that is inside the segment with actual nodes. This is
    happens coz we we are currently doing segmentation on graph that might have
    loops inside eia ops.
    3. Remove all segments in remove list from segment list.
  */
  std::unordered_map<string, tensorflow::Node*> node_map;
  BuildNodeMap(graph, &node_map);
  std::set<int> remove_segment_index;
  for (auto idx = 0; idx < normal_segments.size(); idx++) {
    std::set<std::string>& nodes_in_segment = normal_segments[idx];
    std::set<int> subgraph_node_ids;
    for (auto node_name : nodes_in_segment) {
      subgraph_node_ids.insert(node_map[node_name]->id());
    }

    tensorflow::EdgeSet incoming_edges;
    tensorflow::EdgeSet outgoing_edges;
    GetSubGraphIncomingEdges(graph, subgraph_node_ids, &incoming_edges);
    GetSubGraphOutgoingEdges(graph, subgraph_node_ids, &outgoing_edges);

    for (const tensorflow::Edge* edge : incoming_edges) {
      if (edge->dst()->input_type(edge->dst_input()) == DT_RESOURCE) {
        EILOG(1) << "incoming edge src: " << edge->src()->name()
                 << " dst: " << edge->dst()->name();
        remove_segment_index.insert(idx);
        break;
      }
    }

    for (const tensorflow::Edge* edge : outgoing_edges) {
      if (edge->src()->output_type(edge->src_output()) == DT_RESOURCE) {
        EILOG(1) << "outgoing edge src: " << edge->src()->name()
                 << " dst: " << edge->dst()->name();
        remove_segment_index.insert(idx);
        break;
      }
    }

    // Replace any eia op in segment to its corresponding nodes
    ReplaceLoopEIAOpsWithNodes(nodes_in_segment, loop_segments,
                               eop_index_to_name_map);
  }

  // Removing segment from segemnt list.
  uint remove_segment_count = 0;
  for (auto idx : remove_segment_index) {
    uint rel_idx = idx - remove_segment_count++;
    EILOG(1) << " Removing segment :" << idx
             << " num of segment nodes : " << normal_segments[rel_idx].size();
    normal_segments.erase(normal_segments.begin() + rel_idx);
  }
}

bool HasAttr(const NodeDef& node, const string& attr_name) {
  return node.attr().count(attr_name) > 0;
}

const string& GetStringAttr(const NodeDef& node, const string& attr_name) {
  CHECK(HasAttr(node, attr_name));
  const auto& attr = node.attr().at(attr_name);
  CHECK_EQ(attr.value_case(), AttrValue::kS);
  return attr.s();
}

// Performing :
// 1. Clearing device info from all nodes
// 2. Removing _class:loc attr from nodedef
Status PreProcessingGraphDef(const std::vector<string>& output_names,
                             GraphDef& in_graph_def) {
  EILOG(1) << " Creating Identities for all outputs";

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

  EILOG(2) << in_graph_def.DebugString();

  return tensorflow::Status::OK();
}

// Check if all the loop ops are in whitelist. If not all are in whitelist
// remove the remaining from the list.
bool SanitizeWhiteListforLoopOps(std::set<std::string>* op_whitelist,
                                 std::vector<string>* loop_ops_deleted) {
  std::vector<string> loop_ops{"Enter",         "Merge",    "Switch",
                               "NextIteration", "LoopCond", "Exit"};

  bool IsAllLoopOpsWhiteListed = true;
  for (auto ops : loop_ops) {
    if (op_whitelist->find(ops) == op_whitelist->end()) {
      IsAllLoopOpsWhiteListed = false;
      break;
    }
  }

  if (!IsAllLoopOpsWhiteListed) {
    // Can't have only partial list of loop ops in whitelist, since they
    // always need to run on one device. Removing from whitelist.
    for (auto ops : loop_ops) {
      if (op_whitelist->find(ops) != op_whitelist->end()) {
        op_whitelist->erase(ops);
        loop_ops_deleted->push_back(ops);
      }
    }
  }

  return IsAllLoopOpsWhiteListed;
}

// This does a DFS to find all the ops between EI ops.
void FindCulpritOps(Node* curr_node, std::set<string>& culprit_ops,
                    std::set<string> temp_list,
                    const std::set<string>& input_names_set,
                    const std::set<string>& output_names_set,
                    std::set<string>& processed_nodes,
                    const std::set<std::string>* op_whitelist,
                    const std::set<std::string>* op_cpu,
                    const std::set<std::string>* op_inferentia,
                    bool start_collection) {
  EILOG(1) << " FindCulpritOps: Processing node: " << curr_node->name()
           << " op: " << curr_node->op_def().name();

  auto op_name = curr_node->op_def().name();
  auto node_name = curr_node->name();
  // if we hit output node, then we will clear the temp_list, as these
  // ops are not between EIA ops
  if (output_names_set.count(node_name)) {
    EILOG(1) << "FindCulpritOps : curr node: " << node_name
             << " Is end node returning ..";
    temp_list.clear();
    return;
  }

  // if current node is an EIAOp or a processed node which is already a culprit
  // node, and we have already seen EIA op in its ancestor, add all the ops
  // collected to the final culprit_ops list.
  if (op_name == "InferentiaOp" ||
      (processed_nodes.count(node_name) && culprit_ops.count(op_name))) {
    if (start_collection) {
      culprit_ops.insert(temp_list.begin(), temp_list.end());
      temp_list.clear();
    } else {
      start_collection = true;
    }
  } else {
    // if EIAOp seen before  and this op is not EIA candidate
    // add this op to the temp_list.
    if (start_collection && !IsEIACandidate(
        curr_node, *op_whitelist, *op_cpu, *op_inferentia)) {
      EILOG(1) << "FindCulpritOps: Inserting to list :" << curr_node->name()
               << " Op: " << curr_node->op_def().name();
      temp_list.insert(op_name);
    }
  }

  // If node  _SOURCE/_SINK/Processed Node  then return.
  if (curr_node->IsSource() || curr_node->IsSink() ||
      processed_nodes.count(node_name) > 0) {
    EILOG(1) << "FindCulpritOps: curr op: " << op_name
             << " Is Source/Sink/Processed returning ..";
    return;
  }

  // Processing all out edges
  // if NextIteration node then , we shouldnt process outgoing nodes for this
  // node, as this will cause cycles, but still mark it as Processed.
  if (!curr_node->IsNextIteration()) {
    for (auto edge : curr_node->out_edges()) {
      Node* node = edge->dst();
      FindCulpritOps(node, culprit_ops, temp_list, input_names_set,
                     output_names_set, processed_nodes,
                     op_whitelist, op_cpu, op_inferentia,
                     start_collection);
    }
  }

  processed_nodes.insert(curr_node->name());
}

// To find ops that are causing EI ops to break.
std::set<string> GetOpsBetweenEIOps(tensorflow::Graph& graph,
                                    const std::vector<string>& inputs,
                                    const std::vector<string>& output_names,
                                    const std::set<std::string>* op_whitelist,
                                    const std::set<std::string>* op_cpu,
                                    const std::set<std::string>* op_inferentia) {
  std::unordered_map<string, tensorflow::Node*> node_map;
  std::set<string> processed_nodes;

  std::set<string> output_names_set(output_names.begin(), output_names.end());
  std::set<string> input_names_set(inputs.begin(), inputs.end());
  bool start_collection = false;

  BuildNodeMap(graph, &node_map);
  std::set<string> culprit_ops;
  std::set<string> temp_list;

  for (auto input_name : inputs) {
    Node* start_node = node_map[input_name];
    FindCulpritOps(start_node, culprit_ops, temp_list, input_names_set,
                   output_names_set, processed_nodes,
                   op_whitelist, op_cpu, op_inferentia,
                   start_collection);
  }

  // Iterate through all the nodes, and find unprocessed nodes.
  // If EIAop node,find nodes that might be connected and between other
  // EI ops
  // else add to new list , since we do not know if this will be processed
  // while processing an unprocessed EI node.
  temp_list.clear();
  start_collection = false;
  std::vector<Node*> possible_counted_nodes;
  for (auto node : graph.op_nodes()) {
    auto op_name = node->op_def().name();
    if (processed_nodes.find(node->name()) == processed_nodes.end()) {
      EILOG(1) << " GetOpsBetweenEIOps: Unprocessed nodes: " << op_name;
      if (op_name == "InferentiaOp") {
        FindCulpritOps(node, culprit_ops, temp_list, input_names_set,
                       output_names_set, processed_nodes,
                       op_whitelist, op_cpu, op_inferentia,
                       start_collection);
      } else {
        if (!(IsEIACandidate(node, *op_whitelist, *op_cpu, *op_inferentia))) {
          possible_counted_nodes.push_back(node);
        }
      }
    }
  }

  return culprit_ops;
}

// Returns total number of nodes in all segments.
uint GetNumOfOpsInSegments(kaena::segment::SegmentNodesVector& segments) {
  uint total_num_nodes_in_segments = 0;
  for (auto s : segments) {
    EILOG(2) << "size: " << s.size();
    total_num_nodes_in_segments += s.size();
  }
  return total_num_nodes_in_segments;
}

// This function is the base function which does:
// Step 1: Find EIA Segments.
// Step 2: Calls functions to create EIA subgraphs.
static tensorflow::Status BuildEIAOp(
    GraphDef& in_graph_def, const std::vector<string>& inputs,
    const std::vector<string>& tensor_output_names,
    tensorflow::GraphDef& new_graph_def, const int minimum_segment_size,
    std::set<std::string>* op_whitelist,
    std::set<std::string>* op_cpu, std::set<std::string>* op_inferentia) {
  string model_id("");

  // For storing just names without tensor
  std::vector<string> output_names;
  for (auto name : tensor_output_names) {
    output_names.push_back(split(name, ":")[0]);
  }

  // Segment the graph into subgraphs that can be converted to EIA op
  tensorflow::kaena::segment::SegmentOptions segment_options;

  EILOG(1) << "Building EI Op\n";

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
  uint total_nodes_original_graph = graph.num_nodes();

  std::unordered_map<string, int> eop_index_to_name_map;

  // ############################ STEP 1: #################################
  // Making a graph with  with only loops in EIA ops.

  tensorflow::kaena::segment::SegmentNodesVector loop_segments;

  tensorflow::Graph loop_graph(flib);

  TF_CHECK_OK(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), in_graph_def, &loop_graph));
  int start_eop_index = 0;

  std::vector<string> loop_ops_deleted;
  bool IsAllLoopOpsWhiteListed =
      SanitizeWhiteListforLoopOps(op_whitelist, &loop_ops_deleted);

  // Only do loops segmentation if all the ops in "loop_ops" list
  // are in whitelist.
  std::set<string> loop_breaking_ops;
  if (IsAllLoopOpsWhiteListed) {
    TF_RETURN_IF_ERROR(tensorflow::kaena::segment::SegmentGraphForLoops(
        in_graph_def, IsEIACandidate, &loop_segments,
        segment_options.exclude_node_list, loop_breaking_ops,
        op_whitelist, op_cpu, op_inferentia));
    if (loop_segments.size() > 1) {
      EILOG(1) << "MULTIPLE loop ei candidate conversion: "
               << loop_segments.size();
    }

    // Process segments only if there are any loop segments.
    if (loop_segments.size()) {
      TF_RETURN_IF_ERROR(ProcessSegments(
          loop_graph, tensor_output_names, node_map, loop_segments,
          start_eop_index, &eop_index_to_name_map, model_id));
    }
  }

  // ###################  STEP 2 #########################################
  /// Once the loops are in the graph , we will use that graph to form normal
  /// segments. We will add eia to whitelist , so that they are engulfed if
  /// needed.

  // Adding EIAOp to whitelist so that its consumed while forming eia ops
  // The loop eia ops could  be consumed by another eia  segment.
  // We will replace the loop eia op with all the nodes inside it , while
  // forming the final graph
  op_whitelist->insert("InferentiaOp");

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

  tensorflow::kaena::segment::SegmentNodesVector normal_segments;
  GraphDef graph_def_with_loops;
  loop_graph.ToGraphDef(&graph_def_with_loops);

  TF_RETURN_IF_ERROR(tensorflow::kaena::segment::SegmentGraph(
      graph_def_with_loops, IsEIACandidate, segment_options, &normal_segments,
      op_whitelist, op_cpu, op_inferentia));
  if (normal_segments.size() > 1) {
    EILOG(1) << "MULTIPLE ei candidate conversion: " << normal_segments.size();
  }

  if (normal_segments.size()) {
    PreProcessSegmentsForResources(loop_graph, normal_segments, loop_segments,
                                   &eop_index_to_name_map);

    // Start processing final segments

    // Now work on original graph to make the final eia graph.
    TF_RETURN_IF_ERROR(ProcessSegments(graph, tensor_output_names, node_map,
                                       normal_segments, start_eop_index,
                                       &eop_index_to_name_map, model_id));
  }

  // ################### STEP 2 DONE #######################

  // ##################### STEP 3 ############################

  // On the new graph , we want to send all the nodes that is in white list
  // and has data-format at NCHW, this is mostly needed for pooling ops for
  // which NHWC is the only format supported on CPU
  std::unordered_map<string, tensorflow::Node*> final_node_map;
  TF_RETURN_IF_ERROR(BuildNodeMap(graph, &final_node_map));

  tensorflow::kaena::segment::SegmentNodesVector one_node_NCHW_segments;

  // we dont want to send input and output nodes in final graph, so adding
  // them to an exclude list.
  std::set<string> exclude_names_set(output_names.begin(), output_names.end());
  exclude_names_set.insert(inputs.begin(), inputs.end());
  for (auto node : graph.nodes()) {
    auto node_def = node->def();
    bool hasNCHW = false;
    for (auto const& input_name : node_def.input()) {
      for (const auto& attr : node_def.attr()) {
        if (exclude_names_set.find(node->name()) == exclude_names_set.end() &&
            IsEIACandidate(node, *op_whitelist, *op_cpu, *op_inferentia) &&
            HasAttr(node_def, "data_format")) {
          if (GetStringAttr(node_def, "data_format") == "NCHW") {
            EILOG(1) << " Node with data format: " << node_def.name();
            hasNCHW = true;
          }
        }
      }
    }

    if (hasNCHW) {
      EILOG(1) << "single node segment: " << node_def.name();
      std::set<string> single_item_set = {node_def.name()};
      one_node_NCHW_segments.push_back(single_item_set);
    }
  }

  if (one_node_NCHW_segments.size()) {
    // Still remove ei ops that take/outputs resources
    PreProcessSegmentsForResources(graph, one_node_NCHW_segments, loop_segments,
                                   &eop_index_to_name_map);

    // Now work on original graph to make the final eia graph.
    TF_RETURN_IF_ERROR(ProcessSegments(
        graph, tensor_output_names, final_node_map, one_node_NCHW_segments,
        start_eop_index, &eop_index_to_name_map, model_id));
  }

  // ################## STEP 3 DONE ##########################

  // Reset whitelist. Not needed but doing for sanity.
  op_whitelist->erase("InferentiaOp");

  // Loops ops can be removed from whitelist in 2 places:
  // 1. SanitizeWhiteListforLoopOps()
  // 2. segment.cc -> GetLoopSegments()
  if (IsAllLoopOpsWhiteListed) {
    op_whitelist->insert("Enter");
    op_whitelist->insert("Merge");
    op_whitelist->insert("LoopCond");
    op_whitelist->insert("Exit");
    op_whitelist->insert("Switch");
    op_whitelist->insert("NextIteration");
  } else {
    for (auto ops : loop_ops_deleted) {
      op_whitelist->insert(ops);
    }
  }

  graph.ToGraphDef(&new_graph_def);

  EILOG(2) << "new_graph_def: " << new_graph_def.DebugString();

  return tensorflow::Status::OK();
}

static Status CreateEiaGraphDef(GraphDef& in_graph_def,
                                const std::vector<string>& inputs,
                                const std::vector<string>& outputs,
                                const int minimum_segment_size,
                                std::set<std::string>* op_whitelist,
                                std::set<std::string>* op_cpu,
                                std::set<std::string>* op_inferentia,
                                GraphDef& new_graph_def) {
  OpRegistryInterface* op_registry = OpRegistry::Global();
  // Add default attributes to all nodes in the graph def, This
  // should solve the index_type problem for faster rcnn
  TF_CHECK_OK(AddDefaultAttrsToGraphDef(&in_graph_def, *op_registry, 0, true));

  return BuildEIAOp(in_graph_def, inputs, outputs, new_graph_def,
                    minimum_segment_size, op_whitelist, op_cpu, op_inferentia);
}

// This is the first function that gets called from python (eia_convert)
// linked through swig file eia_conversion.i
Status ConvertGraphDefToEIA(string *new_graph_def_str,
                            const string &graph_def_str,
                            const string &inputs_str,
                            const string &outputs_str,
                            const string &op_whitelist_str,
                            const string &op_cpu_str,
                            const string &op_inferentia_str,
                            const int min_seg_size) {
  tensorflow::Status status = tensorflow::Status::OK();

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
  std::set<std::string> op_cpu;
  temp.ParseFromString(op_cpu_str);
  for (const auto& name : temp.s()) {
    op_cpu.insert(name);
  }
  std::set<std::string> op_inferentia;
  temp.ParseFromString(op_inferentia_str);
  for (const auto& name : temp.s()) {
    op_inferentia.insert(name);
  }
  if (min_seg_size < 1) {
    return tensorflow::errors::InvalidArgument("min_seg_size >= 1 required");
  }

  uint64 start_convert_us = Env::Default()->NowMicros();

  // Debug code
  // EILOG(1) << "_______before______";
  // tensorflow::FunctionLibraryDefinition
  // flib(tensorflow::OpRegistry::Global(),
  //                                            graph_def.library());
  // tensorflow::Graph graph(flib);
  // // TODO: check for error
  // TF_CHECK_OK(tensorflow::ConvertGraphDefToGraph(
  //     tensorflow::GraphConstructorOptions(), graph_def, &graph));
  // printGraphEdges(&graph);
  // EILOG(1) << "_______before______";
  status = CreateEiaGraphDef(graph_def, inputs, outputs, min_seg_size,
                             &op_whitelist, &op_cpu, &op_inferentia, new_graph_def);
  new_graph_def.SerializeToString(new_graph_def_str);
  // Debug code
  // EILOG(1) << "_______after______";
  // tensorflow::FunctionLibraryDefinition
  // flib(tensorflow::OpRegistry::Global(),
  //                                            new_graph_def.library());
  // tensorflow::Graph graph(flib);
  // TF_CHECK_OK(tensorflow::ConvertGraphDefToGraph(
  //     tensorflow::GraphConstructorOptions(), new_graph_def, &graph));
  // printGraphEdges(&graph);
  // EILOG(1) << "_______after______";

  uint64 convert_time_us = Env::Default()->NowMicros() - start_convert_us;
  EILOG(1) << "Conversion Took " << convert_time_us / 1000 << "ms\n";
  return status;
}

}  // namespace convert
}  // namespace kaena
}  // namespace tensorflow

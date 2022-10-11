/* Copyright Amazon Web Services and its Affiliates. All Rights Reserved.

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

#include "tensorflow/neuron/grappler/fuse_supported_operators.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/neuron/grappler/convert/convert_graph.h"
#include "tensorflow/neuron/grappler/graph_constructor_wrapper.h"
#include "tensorflow/neuron/grappler/graph_optimizer_registry.h"

namespace tensorflow {
namespace grappler {
namespace neuron {

namespace {

constexpr char key_minimum_segment_size[] = "minimum_segment_size";
constexpr char key_fuse_foldable_nodes[] = "fuse_foldable_nodes";
constexpr char key_automatic[] = "automatic";
constexpr char key_prune_small_subgraphs_ratio[] =
    "prune_small_subgraphs_ratio";
constexpr char key_supported_op_types[] = "supported_op_types";
constexpr char key_no_fuse_ops[] = "no_fuse_ops";
constexpr char key_force_fuse_ops[] = "force_fuse_ops";
constexpr char kNeuronInferredShapes[] = "_aws_neuron_inferred_shapes";
constexpr int kNeuronFewChannelThreshold = 32;
constexpr char kNeuronOptimizerConfig[] = "_aws_neuron_optimizer_config";
constexpr char kInputOpNames[] = "input_op_names";
constexpr char kOutputOpNames[] = "output_op_names";
constexpr char name_optimizer[] = "aws_neuron_fuse_supported_operators";

template <class T>
std::string container_debug_string(const T& container) {
  std::string debug_string;
  for (const auto& item : container) {
    debug_string += ",";
    debug_string += item;
  }
  return debug_string.substr(0 < debug_string.size() ? 1 : 0);
}

PartialTensorShape ReadInferredShape(const Edge* edge) {
  const Node* node = edge->src();
  const auto& attr = node->def().attr();
  if (!attr.count(kNeuronInferredShapes)) {
    return PartialTensorShape();
  }
  const auto& inferred_shapes = attr.at(kNeuronInferredShapes).list();
  return PartialTensorShape(inferred_shapes.shape(edge->src_output()));
}

Status MaybeExcludeFirstConv2DParents(std::set<std::string>* no_fuse_ops,
                                      const Graph& graph) {
  // Conditions for mapping Conv2D parents to CPU:
  //  1. There is no built-in padding
  //  2. Strides are all greater than 1
  //  3. Strides are all equal
  //  4. Input spatial sizes are all divisible by corresponding strides
  //  5. Number of input channels is small (current threshold is 32)
  //  6. Kernel spatial sizes are all greater than 1
  //  7. Not running group convolution
  //  8. Not running dilated convolution
  Node* first_conv2d = nullptr;
  for (Node* node : graph.nodes()) {
    if (node->type_string() == "Conv2D") {
      first_conv2d = node;
      break;
    }
  }
  std::vector<Node*> starts;
  while (first_conv2d != nullptr) {
    VLOG(1) << "Found Conv2D " << first_conv2d->name();
    const auto& attr = first_conv2d->def().attr();
    if (attr.at("padding").s() != "VALID") break;  // condition 1
    VLOG(3) << "There is no built-in padding";
    const Edge* input_edge;
    TF_RETURN_IF_ERROR(first_conv2d->input_edge(0, &input_edge));
    PartialTensorShape input_shape = ReadInferredShape(input_edge);
    if (!input_shape.IsFullyDefined()) break;
    const std::string data_format = attr.at("data_format").s();
    if (data_format != "NHWC" && data_format != "NCHW") break;
    int dim_h = data_format.find("H");
    int dim_w = data_format.find("W");
    int dim_c = data_format.find("C");
    AttrValue_ListValue strides = attr.at("strides").list();
    int stride_h = strides.i(dim_h);
    int stride_w = strides.i(dim_w);
    if (stride_h <= 1 && stride_w <= 1) break;  // condition 2
    VLOG(3) << "Strides are all greater than 1";
    if (stride_h != stride_w) break;  // condition 3
    VLOG(3) << "Strides are all equal";
    bool h_divisible = (input_shape.dim_size(dim_h) % stride_h) == 0;
    bool w_divisible = (input_shape.dim_size(dim_w) % stride_w) == 0;
    if (!(h_divisible && w_divisible)) break;  // condition 4
    VLOG(3) << "Input spatial sizes are all divisible by corresponding strides";
    int input_channels = input_shape.dim_size(dim_c);
    if (input_channels >= kNeuronFewChannelThreshold) break;  // condition 5
    VLOG(3) << "Number of input channels is small";
    const Edge* filter_edge;
    TF_RETURN_IF_ERROR(first_conv2d->input_edge(1, &filter_edge));
    PartialTensorShape filter_shape = ReadInferredShape(filter_edge);
    if (!filter_shape.IsFullyDefined()) break;
    int window_h = filter_shape.dim_size(0);
    int window_w = filter_shape.dim_size(1);
    int filter_input_channels = filter_shape.dim_size(2);
    if (window_h <= 1 && window_w <= 1) break;  // condition 6
    VLOG(3) << "Kernel spatial sizes are all greater than 1";
    if (input_channels != filter_input_channels) break;  // condition 7
    VLOG(3) << "Not running group convolution";
    AttrValue_ListValue dilations = attr.at("dilations").list();
    int dilation_h = dilations.i(dim_h);
    int dilation_w = dilations.i(dim_w);
    if (dilation_h > 1 || dilation_w > 1) break;  // condition 8
    VLOG(3) << "Not running dilated convolution";
    starts.push_back(input_edge->src());
    break;
  }
  if (!starts.empty()) {
    VLOG(1) << "Excluding Conv2D parents from " << starts.back()->name();
    auto enter = [&](Node* node) {
      if (node->IsOp() && node->type_string() != "Placeholder") {
        no_fuse_ops->emplace(node->name());
      }
    };
    ReverseDFSFrom(graph, starts, enter, /*leave=*/nullptr);
  }
  if (no_fuse_ops->size() > graph.num_op_nodes() / 2) {
    // Don't exclude if need to exclude more than 50% of total ops
    no_fuse_ops->clear();
  }
  return Status::OK();
}

void ExcludeInt64Select(std::set<std::string>* no_fuse_ops,
                        const Graph& graph) {
  for (const Node* node : graph.nodes()) {
    const auto& attr = node->def().attr();
    if (node->type_string() == "Select" && attr.at("T").type() == DT_INT64) {
      no_fuse_ops->emplace(node->name());
    }
  }
}

void ExcludeConcatCheapConsumers(
    std::set<std::string>* no_fuse_ops, const Graph& graph,
    const std::set<std::string>& expensive_op_types) {
  std::vector<Node*> starts;
  for (Node* node : graph.nodes()) {
    if (node->type_string() == "ConcatV2") {
      const auto& attr = node->def().attr();
      DataType dtype(attr.at("T").type());
      int in_degree = attr.at("N").i();
      if (DataTypeIsFloating(dtype) && in_degree > 3) {
        starts.push_back(node);
      }
    }
  }
  std::vector<Node*> consumers;
  auto enter = [&](Node* node) {
    if (node->IsOp()) {
      consumers.push_back(node);
    }
  };
  DFSFrom(graph, starts, enter, /*leave=*/nullptr);
  bool found_expensive_ops = false;
  for (const Node* node : consumers) {
    if (expensive_op_types.count(node->type_string())) {
      found_expensive_ops = true;
      break;
    }
  }
  if (!found_expensive_ops) {
    for (const Node* node : consumers) {
      no_fuse_ops->emplace(node->name());
    }
  }
}

std::set<std::string> ExpensiveTypeStrings() {
  return {
      "BatchMatMul",
      "BatchMatMulV2",
      "BatchMatMulV3",
      "Conv2D",
      "Conv2DBackpropFilter",
      "Conv2DBackpropInput",
      "Conv3D",
      "Conv3DBackpropFilterV2",
      "Conv3DBackpropInputV2",
      "DepthwiseConv2dNative",
      "DepthwiseConv2dNativeBackpropFilter",
      "DepthwiseConv2dNativeBackpropInput",
      "Einsum",
      "MatMul",
  };
}

}  // namespace

Status FuseSupportedOperators::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  return Status::OK();
}

std::string FuseSupportedOperators::name() const { return name_optimizer; }

Status FuseSupportedOperators::Optimize(Cluster* cluster,
                                        const GrapplerItem& item,
                                        GraphDef* output) {
  if (cluster == nullptr) {
    return errors::InvalidArgument("cluster == nullptr");
  }
  VLOG(5) << "item.graph=" << item.graph.DebugString();
  const NodeDef* oc_node = nullptr;
  for (const NodeDef& node : item.graph.node()) {
    if (node.op() == "Placeholder" || node.op() == "PlaceholderWithDefault") {
      if (node.attr().count(kNeuronOptimizerConfig)) {
        oc_node = &node;
        break;
      }
    }
  }
  if (nullptr == oc_node) {
    return errors::InvalidArgument("Did not find optimizer config node.");
  }
  const auto& attr = oc_node->attr().at(kNeuronOptimizerConfig).func().attr();
  std::vector<std::string> input_op_names;
  std::vector<std::string> output_op_names;
  int minimum_segment_size = 1;
  bool fuse_foldable_nodes = false;
  bool automatic = false;
  double prune_small_subgraphs_ratio = 0.0;
  std::set<std::string> supported_op_types;
  std::set<std::string> no_fuse_ops;
  std::set<std::string> force_fuse_ops;
  if (!attr.count(kInputOpNames)) {
    return errors::InvalidArgument(name_optimizer, " requires input op names");
  }
  const auto& attr_input_op_names = attr.at(kInputOpNames).list().s();
  input_op_names = {attr_input_op_names.begin(), attr_input_op_names.end()};
  if (!attr.count(kOutputOpNames)) {
    return errors::InvalidArgument(name_optimizer, " requires output op names");
  }
  const auto& attr_output_op_names = attr.at(kOutputOpNames).list().s();
  output_op_names = {attr_output_op_names.begin(), attr_output_op_names.end()};
  if (attr.count(key_minimum_segment_size)) {
    minimum_segment_size = attr.at(key_minimum_segment_size).i();
  }
  if (attr.count(key_fuse_foldable_nodes)) {
    fuse_foldable_nodes = attr.at(key_fuse_foldable_nodes).b();
  }
  if (attr.count(key_automatic)) {
    automatic = attr.at(key_automatic).b();
  }
  if (attr.count(key_prune_small_subgraphs_ratio)) {
    prune_small_subgraphs_ratio = attr.at(key_prune_small_subgraphs_ratio).f();
  }
  if (!attr.count(key_supported_op_types)) {
    return errors::InvalidArgument(
        name_optimizer,
        " requires providing a list of supported operator names");
  }
  const auto& param_supported_op_types =
      attr.at(key_supported_op_types).list().s();
  supported_op_types = {param_supported_op_types.begin(),
                        param_supported_op_types.end()};
  VLOG(2) << "supported_op_types "
          << container_debug_string(supported_op_types);
  if (attr.count(key_no_fuse_ops)) {
    const auto& param_no_fuse_ops = attr.at(key_no_fuse_ops).list().s();
    no_fuse_ops = {param_no_fuse_ops.begin(), param_no_fuse_ops.end()};
  }
  VLOG(2) << "no_fuse_ops " << container_debug_string(no_fuse_ops);
  if (attr.count(key_force_fuse_ops)) {
    const auto& param_force_fuse_ops = attr.at(key_force_fuse_ops).list().s();
    force_fuse_ops = {param_force_fuse_ops.begin(), param_force_fuse_ops.end()};
  }
  VLOG(2) << "force_fuse_ops " << container_debug_string(force_fuse_ops);

  std::set<std::string> expensive_op_types;
  if (automatic) {
    no_fuse_ops.clear();
    force_fuse_ops.clear();
    FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());
    Graph graph(flib);
    TF_RETURN_IF_ERROR(
        ConvertGraphDefToGraph(GraphConstructorOptions(), item.graph, &graph));

    // Exclude Pad etc. that precede the first few-channel Conv2D
    TF_RETURN_IF_ERROR(MaybeExcludeFirstConv2DParents(&no_fuse_ops, graph));

    // Exclude int64 Select ops
    ExcludeInt64Select(&no_fuse_ops, graph);

    // Exclude arithmetic-unintensive consumers of Concat for object detectors
    expensive_op_types = ExpensiveTypeStrings();
    ExcludeConcatCheapConsumers(&no_fuse_ops, graph, expensive_op_types);

    VLOG(2) << "auto no_fuse_ops " << container_debug_string(no_fuse_ops);
  }
  VLOG(2) << "input_op_names " << container_debug_string(input_op_names);
  VLOG(2) << "output_op_names " << container_debug_string(output_op_names);
  TF_RETURN_IF_ERROR(tensorflow::neuron::convert::CreateNeuronGraphDef(
      output, item.graph, input_op_names, output_op_names, fuse_foldable_nodes,
      minimum_segment_size, prune_small_subgraphs_ratio, supported_op_types,
      no_fuse_ops, force_fuse_ops, expensive_op_types));
  return Status::OK();
}

void FuseSupportedOperators::Feedback(Cluster* cluster,
                                      const GrapplerItem& item,
                                      const GraphDef& optimize_output,
                                      double result) {
  // Nothing to do for FuseSupportedOperators.
}

REGISTER_NEURON_GRAPH_OPTIMIZER_AS(FuseSupportedOperators, name_optimizer);

}  // end namespace neuron
}  // end namespace grappler
}  // end namespace tensorflow

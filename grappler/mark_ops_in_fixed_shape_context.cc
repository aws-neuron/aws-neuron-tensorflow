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

#include <string>
#include <unordered_set>
#include <vector>
#include "absl/algorithm/container.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/neuron/grappler/convert/graph_constructor_wrapper.h"
#include "tensorflow/neuron/grappler/graph_optimizer_registry.h"

namespace tensorflow {
namespace grappler {
namespace neuron {
namespace {

constexpr char kNameOptimizer[] = "aws_neuron_mark_ops_in_fixed_shape_context";
constexpr char kNeuronInferredShapes[] = "_aws_neuron_inferred_shapes";
constexpr char kNeuronInFixedShapeContext[] =
    "_aws_neuron_in_fixed_shape_context";

class MarkOpsInFixedShapeContext : public CustomGraphOptimizer {
 public:
  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config =
                  nullptr) override;
  ~MarkOpsInFixedShapeContext() override {}
  std::string name() const override { return kNameOptimizer; }
  bool UsesFunctionLibrary() const { return true; }
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;
  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result);
};

Status MarkOpsInFixedShapeContext::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  return Status::OK();
}

bool IsFixedShapeDataType(DataType dt) {
  return dt != DT_STRING;
}

Status MarkOpsInFixedShapeContext::Optimize(Cluster* cluster,
                                            const GrapplerItem& item,
                                            GraphDef* output) {
  if (cluster == nullptr) {
    return errors::InvalidArgument("cluster == nullptr");
  }
  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());
  Graph graph(flib);
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(GraphConstructorOptions(), item.graph, &graph));

  // Collect node names with fixed shape outputs
  std::unordered_set<std::string> fixed_shape_node_names;
  for (Node* node : graph.op_nodes()) {
    if (!node->def().attr().count(kNeuronInferredShapes)) {
      continue;
    }
    AttrValue_ListValue inferred_shapes =
        node->def().attr().at(kNeuronInferredShapes).list();
    if (node->type_string() == "FusedBatchNormV3") {
      // FusedBatchNormV3's last output is a temporary buffer used only
      // by FusedBatchNormV3Grad
      inferred_shapes.mutable_shape()->RemoveLast();
    }
    bool fixed_shape = absl::c_all_of(
        inferred_shapes.shape(), [](const TensorShapeProto& shape_proto) {
          return PartialTensorShape(shape_proto).IsFullyDefined();
        });
    fixed_shape &= absl::c_all_of(node->input_types(), IsFixedShapeDataType);
    fixed_shape &= absl::c_all_of(node->output_types(), IsFixedShapeDataType);
    if (fixed_shape) {
      fixed_shape_node_names.insert(node->name());
    }
  }

  // Mark nodes whose all inputs and outputs are fixed shape tensors
  for (Node* node : graph.op_nodes()) {
    bool fixed_shape = absl::c_all_of(node->in_nodes(), [&](Node* in_node) {
      return fixed_shape_node_names.count(in_node->name());
    });
    fixed_shape |= node->num_inputs() == 0;
    VLOG(1) << "Node " << node->name() << " inputs_fixed_shape=" << fixed_shape;
    fixed_shape &= fixed_shape_node_names.count(node->name());
    VLOG(1) << "Node " << node->name() << " fixed_shape=" << fixed_shape;
    node->AddAttr(kNeuronInFixedShapeContext, fixed_shape);
  }
  graph.ToGraphDef(output);
  return Status::OK();
}

void MarkOpsInFixedShapeContext::Feedback(Cluster* cluster,
                                          const GrapplerItem& item,
                                          const GraphDef& optimize_output,
                                          double result) {
  // Nothing to do for MarkOpsInFixedShapeContext.
}

REGISTER_NEURON_GRAPH_OPTIMIZER_AS(MarkOpsInFixedShapeContext, kNameOptimizer);

}  // end namespace
}  // end namespace neuron
}  // end namespace grappler
}  // end namespace tensorflow

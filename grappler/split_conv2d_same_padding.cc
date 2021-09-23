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
#include "tensorflow/neuron/grappler/graph_constructor_wrapper.h"
#include "tensorflow/neuron/grappler/graph_optimizer_registry.h"

namespace tensorflow {
namespace grappler {
namespace neuron {
namespace {

constexpr char kNameOptimizer[] = "aws_neuron_split_conv2d_same_padding";

class SplitConv2DSamePadding: public CustomGraphOptimizer {
 public:
  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config =
                  nullptr) override;
  ~SplitConv2DSamePadding() override {}
  std::string name() const override { return kNameOptimizer; }
  bool UsesFunctionLibrary() const { return true; }
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;
  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;
};

Status SplitConv2DSamePadding::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  return Status::OK();
}

Status SplitConv2DSamePadding::Optimize(Cluster* cluster,
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
    // Determine if there is a Conv2D with same padding
    VLOG(1) << "Node being passed";
    bool first_pass = true;
    NodeDef node_def = node->def();
    if (node_def.op() == "Conv2D") {
      AttrValue& padding_property = node_def->mutable_attr().at("padding");
      std::string padding_str = padding_property.s();

      VLOG(1) << "Found Conv2D Node";
      if (padding_str == "SAME" && first_pass) {
        // 2) Grab a random node, deep copy and clear it out auto_mixed_precision.cc:1081
        NodeDef dummy_node(node->def());
        dummy_node.clear_name(); 
        // 3) Change Conv2D with Same to Valid
        padding_property.set_s("VALID");
        // 4) Add the right padv2 after the convolution
      }
    }


    

    
    /*
    AttrValue_ListValue inferred_shapes =
        node->def().attr().at(kNeuronInferredShapes).list();
    if (node->type_string() == "FusedBatchNormV3") {
      // FusedBatchNormV3's last output is a temporary buffer used only
      // by FusedBatchNormV3Grad
      inferred_shapes.mutable_shape()->RemoveLast();
    }
    bool has_fixed_shape_outputs = absl::c_all_of(
        inferred_shapes.shape(), [](const TensorShapeProto& shape_proto) {
          return PartialTensorShape(shape_proto).IsFullyDefined();
        });
    if (has_fixed_shape_outputs) {
      fixed_shape_node_names.insert(node->name());
    }
    */
  }

  // Mark nodes whose all inputs and outputs are fixed shape tensors
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

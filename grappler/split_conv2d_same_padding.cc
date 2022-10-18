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
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/neuron/grappler/convert/split_conv2d.h"
#include "tensorflow/neuron/grappler/graph_optimizer_registry.h"

namespace tensorflow {
namespace grappler {
namespace neuron {
namespace {

constexpr char kNameOptimizer[] = "aws_neuron_split_conv2d_same_padding";

class SplitConv2DSamePadding : public CustomGraphOptimizer {
 public:
  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config =
                  nullptr) override;
  ~SplitConv2DSamePadding() override {}
  std::string name() const override { return kNameOptimizer; }
  bool UsesFunctionLibrary() const { return true; }
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;
  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result);
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
  return tensorflow::neuron::convert::SplitConv2DSame(output, item.graph);
}

void SplitConv2DSamePadding::Feedback(Cluster* cluster,
                                      const GrapplerItem& item,
                                      const GraphDef& optimize_output,
                                      double result) {
  // Nothing to do for MarkOpsInFixedShapeContext.
}

REGISTER_NEURON_GRAPH_OPTIMIZER_AS(SplitConv2DSamePadding, kNameOptimizer);

}  // end namespace
}  // end namespace neuron
}  // end namespace grappler
}  // end namespace tensorflow

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
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/neuron/grappler/convert/convert_graph.h"
#include "tensorflow/neuron/grappler/graph_optimizer_registry.h"

namespace tensorflow {
namespace grappler {
namespace neuron {

namespace {
constexpr char name_optimizer[] = "aws_neuron_fuse_supported_operators";
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
  return tensorflow::neuron::convert::ConvertToNeuron(output, item.graph);
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

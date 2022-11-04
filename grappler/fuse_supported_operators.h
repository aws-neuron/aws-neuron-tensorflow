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

#ifndef TENSORFLOW_NEURON_GRAPPLER_FUSE_SUPPORTED_OPERATORS_H_
#define TENSORFLOW_NEURON_GRAPPLER_FUSE_SUPPORTED_OPERATORS_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {
namespace neuron {

class FuseSupportedOperators : public CustomGraphOptimizer {
 public:
  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config =
                  nullptr) override;
  ~FuseSupportedOperators() override {}
  std::string name() const override;
  bool UsesFunctionLibrary() const { return true; }
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;
  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result);
};

}  // end namespace neuron
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_NEURON_GRAPPLER_FUSE_SUPPORTED_OPERATORS_H_

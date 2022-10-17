/* Copyright Amazon Web Services and its Affiliates. Reserved.

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

#include "tensorflow/neuron/grappler/static_shape_inference.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/neuron/grappler/convert/shape_inference.h"
#include "tensorflow/neuron/grappler/graph_optimizer_registry.h"

namespace tensorflow {
namespace grappler {
namespace neuron {

Status StaticShapeInference::Init(
    const RewriterConfig_CustomGraphOptimizer* config) {
  return Status::OK();
}

Status StaticShapeInference::Optimize(Cluster* cluster,
                                      const GrapplerItem& item,
                                      GraphDef* output) {
  if (cluster == nullptr) {
    return errors::InvalidArgument("cluster == nullptr");
  }
  return tensorflow::neuron::convert::ShapeInference(output, item.graph);
}

void StaticShapeInference::Feedback(Cluster* cluster, const GrapplerItem& item,
                                    const GraphDef& optimize_output,
                                    double result) {
  // Nothing to do for StaticShapeInference.
}

REGISTER_NEURON_GRAPH_OPTIMIZER_AS(StaticShapeInference,
                                   name_static_shape_inference);

}  // end namespace neuron
}  // end namespace grappler
}  // end namespace tensorflow

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

#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/neuron/grappler/graph_optimizer_registry.h"
#include "tensorflow/neuron/grappler/static_shape_inference.h"

namespace tensorflow {
namespace grappler {
namespace neuron {

const char kNeuronInferredShapes[] = "_aws_neuron_inferred_shapes";

Status StaticShapeInference::Init(const RewriterConfig_CustomGraphOptimizer *config) {
    return Status::OK();
}

Status PerformStaticShapeInferenceBeforeEncapsulation(Graph *graph) {
    // Perform shape inference.
    std::map<int, InferredShape> arg_shapes;
    GraphShapeInfo shape_info;
    TF_RETURN_IF_ERROR(InferShapes(graph, arg_shapes, /*fnlib_def=*/nullptr, &shape_info));

    // Add attribute for output shapes.
    auto node_name_index = graph->BuildNodeNameIndex();
    for (auto iter : shape_info) {
        std::vector<PartialTensorShape> output_shapes;
        std::transform(
            iter.second.begin(), iter.second.end(), std::back_inserter(output_shapes),
            [](const InferredShape &inferred_shape) {
                return inferred_shape.shape;
            }
        );
        Node *node = node_name_index[iter.first];
        node->ClearAttr(kNeuronInferredShapes);
        node->AddAttr(kNeuronInferredShapes, output_shapes);
    }
    return Status::OK();
}

Status StaticShapeInference::Optimize(Cluster *cluster, const GrapplerItem &item,
                                      GraphDef *output) {
    if (cluster == nullptr) {
        return errors::InvalidArgument("cluster == nullptr");
    }
    std::unique_ptr<GrapplerItem> optimized_item(new GrapplerItem(item));
    std::unique_ptr<Graph> graph = std::unique_ptr<Graph>(new Graph(OpRegistry::Global()));
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, optimized_item->graph, graph.get()));
    TF_RETURN_IF_ERROR(PerformStaticShapeInferenceBeforeEncapsulation(graph.get()));
    graph->ToGraphDef(output);
    return Status::OK();
}

void StaticShapeInference::Feedback(Cluster *cluster, const GrapplerItem &item,
                                    const GraphDef &optimize_output, double result) {
    // Nothing to do for StaticShapeInference.
}

REGISTER_NEURON_GRAPH_OPTIMIZER_AS(StaticShapeInference, name_static_shape_inference);

}  // end namespace neuron
}  // end namespace grappler
}  // end namespace tensorflow

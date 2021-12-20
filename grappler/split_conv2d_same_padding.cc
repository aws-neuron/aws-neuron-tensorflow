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
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/neuron/grappler/graph_constructor_wrapper.h"
#include "tensorflow/neuron/grappler/graph_optimizer_registry.h"

namespace tensorflow {
namespace grappler {
namespace neuron {
namespace {

constexpr char kNameOptimizer[] = "aws_neuron_split_conv2d_same_padding";
constexpr char kNeuronInferredShapes[] = "_aws_neuron_inferred_shapes";

NodeDef MakeNodeConst(const string& name) {
  NodeDef n;
  n.set_name(name);
  n.set_op("Const");
  return n;
}

AttrValue TypeAttrValue(DataType type) {
  AttrValue attr_value;
  SetAttrValue(type, &attr_value);
  return attr_value;
}

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

  // Mark nodes whose all inputs and outputs are fixed shape tensors
	std::string conv2d_original_input = "";
  bool found_conv2d = false;
  int conv2d_idx = 0;
	int copy_node_idx = -1;
  graph.ToGraphDef(output);
  VLOG(1) << "Graph def pre runthrough: " << output->DebugString();
  
  /*
  for (int idx = 0; idx < output->node_size(); idx++) {

    // Change Conv2D with Same to Valid
    NodeDef* node_def = output->mutable_node(idx);

    // TODO: check for case when padding is already valid
    if (node_def->op() == "Conv2D" && !found_conv2d) {
      (*node_def->mutable_attr())["padding"].set_s("VALID");
      found_conv2d = true;
      // Grabs the first conv2d input which is the tensor being operated on
			conv2d_original_input = node_def->input(0);
      conv2d_idx = idx;

      // Grab values to calculate for padding
      AttrValue_ListValue inferred_shapes = node_def->attr().at(kNeuronInferredShapes).list();

      for (const int i : inferred_shapes) {
        VLOG(1) << "Conv shape: " << i;
      }
      AttrValue_ListValue inferred_shapes = node_def().attr().at("strides").list();
      for (const int i : inferred_shapes) {
        VLOG(1) << "Conv shape: " << i;
      }
    }

    if (node_def->op() == "Const" && (*node_def->mutable_attr())["value"].has_tensor()) {
      copy_node_idx = idx;
    }

   
  }

	if (found_conv2d) {
    // Determine input tensor type
    DataType src_type = DT_INVALID;

    for (int idx = 0; idx < output->node_size(); idx++) {

      NodeDef* node_def = output->mutable_node(idx);
      if (node_def->name() == conv2d_original_input) {
        src_type = (*node_def->mutable_attr())["dtype"].type();
      }
    }

		// Adding padding requires two nodes: padding and the values of padding
		// Add operator for value of padding
		const std::string& pad_name = "conv_padding";
    const std::string& pad_op_name = "conv_op_padding";
		if (copy_node_idx == -1) {
			VLOG(1) << "This grappler pass may fail because a pre-allocated tensor was not found";
		}
		else {
			NodeDef x(output->node(copy_node_idx));
			x.clear_name();
			x.set_name(pad_name);
			x.clear_op();
			x.set_op("Const");
			x.clear_device();
			x.set_device(output->node(0).device());
			(*x.mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);

			//Set values of padding array
			auto* mutable_tensor = (*x.mutable_attr())["value"].mutable_tensor();



      // TODO: Remove the hard-coded constants
      std::vector<int> values{0, 0, 0, 0, 0, 0, 0, 0};
      mutable_tensor->set_tensor_content(
              std::string(reinterpret_cast<const char*>(values.data()),
                                  values.size() * sizeof(int)));
      mutable_tensor->set_dtype(DT_INT32);
			auto* tensor_shape = mutable_tensor->mutable_tensor_shape();
			tensor_shape->clear_dim();
			auto tensor_shape_dim0 = tensor_shape->add_dim();
			tensor_shape_dim0->set_size(4);
			
			auto tensor_shape_dim1 = tensor_shape->add_dim();
			tensor_shape_dim1->set_size(2);

			(*output->add_node()) = x;
		}


		// Add operator for pad and set inputs to original Conv2D input and pad value tensor
		NodeDef y(output->node(0));
		y.clear_name();
		y.set_name(pad_op_name);
		y.clear_op();
		y.set_op("Pad");
		y.clear_device();
		y.set_device(output->node(0).device());
		y.clear_input();
    y.add_input();
		y.set_input(0, conv2d_original_input);
    y.add_input();
    y.set_input(1, pad_name);
		y.clear_attr();
		(*y.mutable_attr())["T"] = TypeAttrValue(src_type);
		(*y.mutable_attr())["Tpaddings"] = TypeAttrValue(DT_INT32);
		(*output->add_node()) = y;

    // Rewire input of Conv 2D to be the pad
    NodeDef* conv2d_node_def = output->mutable_node(conv2d_idx);
    conv2d_node_def->set_input(0, pad_name);
	  VLOG(1) << "Neuron graphdef post runthrough: " << output->DebugString();
	}
  */
  return Status::OK();
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

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
#include <iterator>
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
                const GraphDef& optimize_output, double result);
};

Status SplitConv2DSamePadding::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  return Status::OK();
}

std::vector<int> CalculateSamePadding(int input_h, 
                                      int input_w, 
                                      std::vector<int>& strides,
                                      int filter_h,
                                      int filter_w) {
  // Calculations from: https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python 
  int pad_height = 0;
  int pad_width = 0;

  // TODO: Confirm strides are [height, width]. 
  if (strides.size() < 4) {
    VLOG(1) << "Stride size is invalid";
    return std::vector<int>();
  }
  
  if (input_h % strides[0] == 0) {
    pad_height = std::max((filter_h - strides[0]), 0);
  }
  else {
    pad_height = std::max((filter_h - (input_h % strides[0])), 0);
  }

  
  if (input_w % strides[2] == 0) {
    pad_width = std::max((filter_w - strides[2]), 0);
  }
  else {
    pad_width = std::max((filter_w - (input_w % strides[2])), 0);
  }

  int pad_top = pad_height / 2;
  int pad_bottom = pad_height - pad_top;
  int pad_left = pad_width / 2;
  int pad_right = pad_width - pad_left;

  std::vector<int> padding = {pad_top, pad_bottom, pad_left, pad_right};
  return padding;
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
  std::string data_format = "";

  // Variables to store the information for padding
  int filter_w = 0, filter_h = 0, input_w = 0, input_h = 0;
  std::vector<int> stride_vec;

  // Process the graph and find the convolution and the filter
  for (int idx = 0; idx < output->node_size(); idx++) {

    // Change Conv2D with Same to Valid
    NodeDef* node_def = output->mutable_node(idx);

    if (node_def->op() == "Conv2D" && !found_conv2d) {

      // Padding is already valid so we can choose to ignore
      if(node_def->attr().at("padding").s() == "VALID") { 
        VLOG(1) << "Padding optimization is already valid.";
      }
      else {
        found_conv2d = true;
        (*node_def->mutable_attr())["padding"].set_s("VALID");
        // Grabs the first conv2d input which is the tensor being operated on
        // Also store data format for padding order
        data_format = node_def->attr().at("data_format").s();
        conv2d_original_input = node_def->input(0);
        conv2d_idx = idx;

        // Grab value for stride for padding calculation
        AttrValue_ListValue strides = node_def->attr().at("strides").list();
        for (int i = 0; i < strides.i_size(); i++) { 
          stride_vec.push_back(strides.i(i));
        }
      }
    }

    // Grab values from filter for padding calculation
    if (node_def->name() == "Conv2D/filter") {
      AttrValue_ListValue filter_shapes = node_def->attr().at(kNeuronInferredShapes).list();
      const TensorShapeProto shape = filter_shapes.shape(0);

      int HEIGHT_INDEX = 0;
      int WIDTH_INDEX = 1;
      filter_w = shape.dim(HEIGHT_INDEX).size();
      filter_h = shape.dim(WIDTH_INDEX).size();
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
        
        // Find input values
        AttrValue_ListValue input_shapes = node_def->attr().at(kNeuronInferredShapes).list();
        const TensorShapeProto shape = input_shapes.shape(0);
        // Dims are ordered as [batch, depth, width, height] 
        int HEIGHT_INDEX = 2;
        int WIDTH_INDEX = 3;
        input_h = shape.dim(HEIGHT_INDEX).size();
        input_w = shape.dim(WIDTH_INDEX).size();
      }
    }

		// Adding padding requires two nodes: padding and the values of padding
		// Step 1: Add operator for value of padding
		const std::string& pad_name = "Pad/paddings";
    const std::string& pad_op_name = "Pad";
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
      AttrValue::ListValue* mutable_inferred_list = (*x.mutable_attr())["_aws_neuron_inferred_shapes"].mutable_list();

      std::vector<int> padding_constants = CalculateSamePadding(input_h, input_w, stride_vec, filter_h, filter_w);



      std::vector<int> t_values = {0, 0, 0, 0, 0, 0, 0, 0};
      if (data_format == "NCHW") {
        // Offset since N/C have no current padding values
        int PADDING_OFFSET = 4;
        for(int i = 0; i < t_values.size(); i++) {
          if (i >= PADDING_OFFSET) {
            t_values[i] = padding_constants.at(i - PADDING_OFFSET);
          }
        }
      }
      else {
        // Offset: Only code the middle four values since N/C have no padding values
        int PADDING_OFFSET = 2;
        for(int i = 0; i < padding_constants.size(); i++) {
          if (i >= PADDING_OFFSET && i < (t_values.size() - PADDING_OFFSET)) {
            t_values[i] = padding_constants.at(i - PADDING_OFFSET);
          }
        }
      }

      mutable_tensor->set_tensor_content(
              std::string(reinterpret_cast<const char*>(t_values.data()),
                                  t_values.size() * sizeof(int)));
      mutable_tensor->set_dtype(DT_INT32);
			auto* tensor_shape = mutable_tensor->mutable_tensor_shape();
			tensor_shape->clear_dim();
			auto tensor_shape_dim0 = tensor_shape->add_dim();
			tensor_shape_dim0->set_size(4);
			
			auto tensor_shape_dim1 = tensor_shape->add_dim();
			tensor_shape_dim1->set_size(2);

      // Fix the inferred list
      mutable_inferred_list->clear_shape();
      auto* inferred_shape = mutable_inferred_list->add_shape();
      auto* inferred_shape_dim0 = inferred_shape->add_dim();
      inferred_shape_dim0->set_size(4);
      auto* inferred_shape_dim1 = inferred_shape->add_dim();
      inferred_shape_dim1->set_size(2);

			(*output->add_node()) = x;
		}

		// Step 2: Add operator for pad and set inputs to original Conv2D input and pad value tensor
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
    conv2d_node_def->set_input(0, pad_op_name);
	}
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

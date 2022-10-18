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
#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/algorithm/container.h"
#include "graph_constructor_wrapper.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace neuron {
namespace convert {

namespace {

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

std::vector<int> CalculateSamePadding(int input_h, int input_w,
                                      std::vector<int>& strides, int filter_h,
                                      int filter_w, int stride_h_idx,
                                      int stride_w_idx) {
  // Calculations from:
  // https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
  int pad_height = 0;
  int pad_width = 0;

  // Stride indices are determined by data format
  if (strides.size() < 4) {
    VLOG(1) << "Stride size is invalid";
    return std::vector<int>();
  }

  int first_stride = strides[stride_h_idx];
  int second_stride = strides[stride_w_idx];

  if (input_h % first_stride == 0) {
    pad_height = std::max((filter_h - first_stride), 0);
  } else {
    pad_height = std::max((filter_h - (input_h % first_stride)), 0);
  }

  if (input_w % second_stride == 0) {
    pad_width = std::max((filter_w - second_stride), 0);
  } else {
    pad_width = std::max((filter_w - (input_w % second_stride)), 0);
  }

  int pad_top = pad_height / 2;
  int pad_bottom = pad_height - pad_top;
  int pad_left = pad_width / 2;
  int pad_right = pad_width - pad_left;

  std::vector<int> padding = {pad_top, pad_bottom, pad_left, pad_right};
  return padding;
}

}  // end namespace

Status SplitConv2DSame(GraphDef* new_graph_def, const GraphDef& graph_def) {
  FunctionLibraryDefinition flib(OpRegistry::Global(), graph_def.library());
  Graph graph(flib);
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(GraphConstructorOptions(), graph_def, &graph));

  // Mark nodes whose all inputs and outputs are fixed shape tensors
  NodeDef* conv2d_node_def = nullptr;
  std::string conv2d_original_input = "";
  std::string conv2d_filter_input = "";
  bool found_first_conv2d = false;
  bool found_conv2d = false;
  int copy_node_idx = -1;
  graph.ToGraphDef(new_graph_def);
  std::string data_format = "";

  // Variables to store the information for padding
  int filter_w = 0, filter_h = 0, input_w = 0, input_h = 0;
  std::vector<int> stride_vec;

  // Process the graph and find the convolution and the filter
  for (int idx = 0; idx < new_graph_def->node_size(); idx++) {
    // Change the first Conv2D with Same to Valid.
    // This optimization only applies to the very first, 3-channel Conv2D.
    NodeDef* node_def = new_graph_def->mutable_node(idx);

    if (node_def->op() == "Conv2D" && !found_first_conv2d) {
      found_first_conv2d = true;
      if (node_def->attr().at("padding").s() == "SAME") {
        conv2d_node_def = node_def;
        found_conv2d = true;
        // Grabs the first conv2d input which is the tensor being operated on
        // Also store data format for padding order
        data_format = node_def->attr().at("data_format").s();
        conv2d_original_input = node_def->input(0);
        conv2d_filter_input = node_def->input(1);

        // Grab value for stride for padding calculation
        AttrValue_ListValue strides = node_def->attr().at("strides").list();
        for (int i = 0; i < strides.i_size(); i++) {
          stride_vec.push_back(strides.i(i));
        }
      }
    }

    // Grab a random large Const node as an "allocation template" for the new
    // Const node that we'll create to hold padding values.
    // TODO: remove this hack once we solve the protobuf linker issue
    // (migrating to C-API integration is supposed to help)
    if (node_def->op() == "Const" && node_def->ByteSize() > 1024 &&
        (*node_def->mutable_attr())["value"].has_tensor()) {
      copy_node_idx = idx;
    }
  }
  if (copy_node_idx == -1) {
    VLOG(1) << "This grappler pass may fail because a pre-allocated tensor "
               "was not found";
    return Status::OK();
  }

  for (const NodeDef& node : new_graph_def->node()) {
    if (node.name() == conv2d_original_input) {
      const auto& attr = node.attr();
      if (!(attr.count("dtype") || attr.count("T"))) {
        VLOG(1) << "Cannot read data type from node " << node.DebugString();
        return Status::OK();
      }
      break;
    }
  }

  if (found_conv2d) {
    // Determine input tensor type
    DataType src_type = DT_INVALID;

    for (int idx = 0; idx < new_graph_def->node_size(); idx++) {
      NodeDef* node_def = new_graph_def->mutable_node(idx);

      // Pull information from the filter op
      if (node_def->name() == conv2d_filter_input) {
        AttrValue_ListValue filter_shapes =
            node_def->attr().at(kNeuronInferredShapes).list();
        const TensorShapeProto shape = filter_shapes.shape(0);

        int HEIGHT_INDEX = 0;
        int WIDTH_INDEX = 1;
        filter_w = shape.dim(HEIGHT_INDEX).size();
        filter_h = shape.dim(WIDTH_INDEX).size();
      }

      // Pull information from the conv2d op
      if (node_def->name() == conv2d_original_input) {
        const auto& attr = node_def->attr();
        if (attr.count("dtype")) {
          src_type = attr.at("dtype").type();
        } else if (attr.count("T")) {
          src_type = attr.at("T").type();
        } else {
          return errors::InvalidArgument(
              "Cannot read data type from node ", node_def->DebugString());
        }
        VLOG(5) << "conv2d_original_input: " << node_def->DebugString();

        // Find input values
        AttrValue_ListValue input_shapes =
            node_def->attr().at(kNeuronInferredShapes).list();
        const TensorShapeProto shape = input_shapes.shape(0);
        // Dims are ordered as [batch, depth, width, height]
        int HEIGHT_INDEX = data_format.find("H");
        int WIDTH_INDEX = data_format.find("W");
        input_h = shape.dim(HEIGHT_INDEX).size();
        input_w = shape.dim(WIDTH_INDEX).size();
      }
    }

    // Adding padding requires two nodes: padding and the values of padding
    // Step 1: Add operator for value of padding
    const std::string& conv2d_name = conv2d_node_def->name();
    // Use graph.NewName to prevent name clash with existing ops
    std::string pad_op_name = graph.NewName(conv2d_name + "/SamePad");
    std::string pad_name = graph.NewName(pad_op_name + "/paddings");
    if (copy_node_idx == -1) {
      return Status::OK();
    } else {
      NodeDef x(new_graph_def->node(copy_node_idx));
      x.clear_name();
      x.set_name(pad_name);
      x.clear_op();
      x.set_op("Const");
      x.clear_device();
      (*x.mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);

      // Set values of padding array
      auto* mutable_tensor = (*x.mutable_attr())["value"].mutable_tensor();
      AttrValue::ListValue* mutable_inferred_list =
          (*x.mutable_attr())[kNeuronInferredShapes].mutable_list();

      int STRIDE_HEIGHT_IDX = data_format.find("H");
      int STRIDE_WIDTH_IDX = data_format.find("W");
      std::vector<int> padding_constants =
          CalculateSamePadding(input_h, input_w, stride_vec, filter_h, filter_w,
                               STRIDE_HEIGHT_IDX, STRIDE_WIDTH_IDX);

      // We use this because the padding constants needs to be exactly 8 values
      // We then insert the padding constants as needed according to
      // height/width

      std::vector<int> t_values = {0, 0, 0, 0, 0, 0, 0, 0};
      if (data_format == "NCHW") {
        // Offset since N/C have no current padding values
        int PADDING_OFFSET = 4;
        for (int i = 0; i < t_values.size(); i++) {
          if (i >= PADDING_OFFSET) {
            t_values[i] = padding_constants[i - PADDING_OFFSET];
          }
        }
      } else {
        // Offset: Only code the middle four values since N/C have no padding
        // values
        int PADDING_OFFSET = 2;
        for (int i = 0; i < t_values.size(); i++) {
          if (i >= PADDING_OFFSET && i < (t_values.size() - PADDING_OFFSET)) {
            t_values[i] = padding_constants[i - PADDING_OFFSET];
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

      (*new_graph_def->add_node()) = x;
    }

    // Step 2: Add operator for pad and set inputs to original Conv2D input and
    // pad value tensor
    NodeDef y(*conv2d_node_def);
    y.clear_name();
    y.set_name(pad_op_name);
    y.clear_op();
    y.set_op("Pad");
    y.clear_device();
    y.clear_input();
    y.add_input(conv2d_original_input);
    y.add_input(pad_name);
    y.clear_attr();
    (*y.mutable_attr())["T"] = TypeAttrValue(src_type);
    (*y.mutable_attr())["Tpaddings"] = TypeAttrValue(DT_INT32);
    (*new_graph_def->add_node()) = y;

    // Modify Conv2D's padding mode
    conv2d_node_def->mutable_attr()->at("padding").set_s("VALID");

    // Rewire input of Conv 2D to be the pad
    conv2d_node_def->set_input(0, pad_op_name);
  }
  return Status::OK();
}

}  // end namespace convert
}  // end namespace neuron
}  // end namespace tensorflow

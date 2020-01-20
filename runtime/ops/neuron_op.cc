/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


namespace tensorflow {

static Status NeuronOpShape(shape_inference::InferenceContext *ctx) {
    std::vector<int> input_batch_axis;
    TF_RETURN_IF_ERROR(ctx->GetAttr("input_batch_axis", &input_batch_axis));
    bool dynamic_batch_size = false;
    for (auto &axis : input_batch_axis) {
        if (-1 != axis) {
            dynamic_batch_size = true;
            break;
        }
    }
    std::vector<PartialTensorShape> output_shapes;
    TF_RETURN_IF_ERROR(ctx->GetAttr("output_shapes", &output_shapes));
    if (output_shapes.size() != ctx->num_outputs()) {
        VLOG(1) << "Found invalid NodeDef; skipping shape inference";
        return Status::OK();
    }
    for (int idx = 0; idx < ctx->num_outputs(); ++idx) {
        TensorShapeProto shape_proto;
        output_shapes[idx].AsProto(&shape_proto);
        if (dynamic_batch_size) {
            if (idx >= input_batch_axis.size()) {
                VLOG(1) << "input_batch_axis[" << idx << "] is an out-of-bound"
                        << " access; falling back to fixed shape";
            } else {
                int axis = input_batch_axis[idx];
                if (axis < 0 || axis >= shape_proto.dim_size()) {
                    VLOG(1) << "input_batch_axis[" << idx << "] = " << axis
                            << " goes out-of-bound for tensor " << idx
                            << "'s shape; falling back to fixed shape";
                } else {
                    shape_proto.mutable_dim(axis)->set_size(
                        shape_inference::InferenceContext::kUnknownDim);
                }
            }
        }
        PartialTensorShape shape(shape_proto);
        shape_inference::ShapeHandle handle;
        TF_RETURN_IF_ERROR(ctx->MakeShapeFromPartialTensorShape(shape, &handle));
        ctx->set_output(idx, handle);
    }
    return Status::OK();
}

REGISTER_OP("NeuronOp")
    .SetIsStateful()
    .Attr("graph_def: string")
    .Attr("input_names: list(string)")
    .Attr("input_dtypes: list(type) >= 0")
    .Attr("input_shapes: list(shape)")
    .Attr("output_names: list(string)")
    .Attr("output_dtypes: list(type) >= 0")
    .Attr("output_shapes: list(shape)")
    .Attr("executable: string = ''")
    .Attr("input_batch_axis: list(int) = []")
    .Attr("output_batch_axis: list(int) = []")
    .Attr("model_config: list(int) = []")
    .Input("input_tensors: input_dtypes")
    .Output("output_tensors: output_dtypes")
    .SetShapeFn(NeuronOpShape);

} // namespace tensorflow

// model_config format:
//   [global_opt_num_cores, this_opt_num_cores, opt_num_infer, timeout]

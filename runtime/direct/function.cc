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

#include "function.h"

#include <cstddef>
#include <vector>

#include "macros.h"
#include "routine.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/default/logging.h"

namespace tensorflow {
namespace neuron {

Status NeuronFunction::Run(OpKernelContext* ctx, const NodeDef& node_def) {
  TF_RETURN_IF_ERROR(routine_.MaybeInit(node_def, ctx->session_handle()));
  std::vector<Tensor> inputs;
  TF_RETURN_IF_ERROR(SetupInputs(ctx, node_def, &inputs));
  NeuronBatchSharder sharder;
  TF_RETURN_IF_ERROR(sharder.Setup(routine_.Info(), inputs));
  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(SetupOutputs(ctx, node_def, sharder, &outputs));
  std::vector<Tensor> buffers;
  TF_RETURN_IF_ERROR(SetupShuffleBuffers(ctx, inputs, &buffers));
  return routine_.MaybeShardedRun(&sharder, &inputs, &buffers, &outputs);
}

Status NeuronFunction::SetupInputs(OpKernelContext* ctx,
                                   const NodeDef& node_def,
                                   std::vector<Tensor>* inputs) {
  int expected_num_inputs = routine_.Info().input_dtypes.type_size();
  int num_inputs = ctx->num_inputs();
  if (TF_PREDICT_FALSE(num_inputs != expected_num_inputs)) {
    return errors::InvalidArgument("Invalid number of inputs ", num_inputs,
                                   " on NodeDef \"", node_def.name(),
                                   "\" (expect ", expected_num_inputs, ").");
  }

  // Set inputs
  inputs->reserve(num_inputs);
  for (int idx = 0; idx < num_inputs; ++idx) {
    inputs->push_back(ctx->input(idx));
  }
  for (const Tensor& tensor : *inputs) {
    if (TF_PREDICT_FALSE(!DataTypeCanUseMemcpy(tensor.dtype()))) {
      return errors::InvalidArgument("Input tensor ",
                                     tensor.DeviceSafeDebugString(),
                                     " has unsupported data type");
    }
  }
  VLOG(1) << "NeuronFunction::SetupInputs done";
  return Status::OK();
}

Status NeuronFunction::SetupOutputs(OpKernelContext* ctx,
                                    const NodeDef& node_def,
                                    const NeuronBatchSharder& sharder,
                                    std::vector<Tensor>* outputs) {
  int expected_num_outputs = routine_.Info().output_dtypes.type_size();
  int num_outputs = ctx->num_outputs();
  if (TF_PREDICT_FALSE(num_outputs != expected_num_outputs)) {
    return errors::InvalidArgument("Invalid number of outputs ", num_outputs,
                                   " on NodeDef \"", node_def.name(),
                                   "\" (expect ", expected_num_outputs, ").");
  }

  // Allocate and set outputs
  std::vector<Tensor*> output_tensor_ptrs(num_outputs);
  for (int idx = 0; idx < num_outputs; ++idx) {
    const TensorShape& shape = sharder.GetClientOutputShapes().at(idx);
    Tensor** output_ptr_ptr = &output_tensor_ptrs.at(idx);
    TF_RETURN_IF_ERROR(ctx->allocate_output(idx, shape, output_ptr_ptr));
  }
  outputs->reserve(num_outputs);
  for (Tensor* ptr : output_tensor_ptrs) {
    outputs->push_back(*ptr);
  }
  for (const Tensor& tensor : *outputs) {
    if (TF_PREDICT_FALSE(!DataTypeCanUseMemcpy(tensor.dtype()))) {
      return errors::InvalidArgument("Output tensor ",
                                     tensor.DeviceSafeDebugString(),
                                     " has unsupported data type");
    }
  }
  VLOG(1) << "NeuronFunction::SetupOutputs done";
  return Status::OK();
}

Status NeuronFunction::SetupShuffleBuffers(OpKernelContext* ctx,
                                           const std::vector<Tensor>& inputs,
                                           std::vector<Tensor>* buffers) {
  buffers->resize(inputs.size());
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    const Tensor& src = inputs.at(idx);
    Tensor* dst = &buffers->at(idx);
    TF_RETURN_IF_ERROR(ctx->allocate_temp(src.dtype(), src.shape(), dst));
  }
  VLOG(1) << "NeuronFunction::SetupShuffleBuffers done";
  return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow

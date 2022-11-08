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

#include "dynamic_batch.h"
#include <algorithm>
#include <cstddef>
#include <unordered_set>
#include <vector>
#include "executable_info.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/default/logging.h"

namespace tensorflow {
namespace neuron {

static bool IsValidBatchAxis(int batch_axis) { return batch_axis <= 0; }

static bool BatchAxisIsDynamic(int batch_axis) { return batch_axis == 0; }

Status NeuronBatchSharder::Setup(const NeuronExecutableInfo& info,
                                 const std::vector<TensorShape>& input_shapes) {
  // Batch axis argument validity checking
  for (int idx = 0; idx < info.input_batch_axis.i_size(); ++idx) {
    int batch_axis = info.input_batch_axis.i(idx);
    if (TF_PREDICT_FALSE(!IsValidBatchAxis(batch_axis))) {
      return errors::InvalidArgument("Input #", idx, " has invalid batch axis ",
                                     batch_axis);
    }
  }
  for (int idx = 0; idx < info.output_batch_axis.i_size(); ++idx) {
    int batch_axis = info.output_batch_axis.i(idx);
    if (TF_PREDICT_FALSE(!IsValidBatchAxis(batch_axis))) {
      return errors::InvalidArgument("Output #", idx,
                                     " has invalid batch axis ", batch_axis);
    }
  }

  // Initialize client output shapes to NEFF output shapes
  int num_inputs = info.input_shapes.shape_size();
  int num_outputs = info.output_shapes.shape_size();
  client_output_shapes_.reserve(num_outputs);
  for (const TensorShapeProto& neff_shape_proto : info.output_shapes.shape()) {
    TF_RETURN_IF_ERROR(TensorShape::IsValidShape(neff_shape_proto));
    client_output_shapes_.emplace_back(neff_shape_proto);
  }

  // Initialize input/output need-sharding markers to false
  inputs_need_sharding_.resize(num_inputs, false);
  outputs_need_sharding_.resize(num_outputs, false);

  // Read client/NEFF batch size candidates from inputs/input_shapes
  std::unordered_set<int> client_batch_size_set;
  std::unordered_set<int> neff_batch_size_set;
  for (int idx = 0; idx < num_inputs; ++idx) {
    int batch_axis = info.input_batch_axis.i(idx);
    if (!BatchAxisIsDynamic(batch_axis)) {
      continue;
    }

    // Client/NEFF batch sizes
    const TensorShape& input_shape = input_shapes.at(idx);
    const TensorShapeProto& neff_shape_proto = info.input_shapes.shape(idx);
    TF_RETURN_IF_ERROR(TensorShape::IsValidShape(neff_shape_proto));
    TensorShape neff_shape(neff_shape_proto);
    if (input_shape == neff_shape) {
      continue;
    }
    if (TF_PREDICT_FALSE(input_shape.dims() <= batch_axis + 1)) {
      return errors::InvalidArgument(
          "Input tensor #", idx, " has dynamic batch axis ", batch_axis,
          ", but it only has ", input_shape.dims(), " dimensions");
    }
    if (TF_PREDICT_FALSE(neff_shape.dims() <= batch_axis + 1)) {
      return errors::InvalidArgument(
          "NEFF input tensor #", idx, " has dynamic batch axis ", batch_axis,
          ", but it only has ", neff_shape.dims(), " dimensions");
    }
    client_batch_size_set.insert(input_shape.dim_size(batch_axis));
    neff_batch_size_set.insert(neff_shape.dim_size(batch_axis));
    inputs_need_sharding_.at(idx) = true;
  }

  // client_batch_size_set is empty; everything is supposed to have fixed shape
  can_skip_ = client_batch_size_set.empty();
  if (can_skip_) {
    VLOG(1) << "NeuronBatchSharder::Setup done without any dynamic batch size";
    return Status::OK();
  }
  if (TF_PREDICT_FALSE(client_batch_size_set.size() > 1)) {
    return errors::InvalidArgument("Inconsistent client batch sizes");
  }
  if (TF_PREDICT_FALSE(neff_batch_size_set.size() != 1)) {
    return errors::InvalidArgument("Inconsistent NEFF batch sizes");
  }

  // Set has only one element; use it as the client/NEFF batch size
  client_batch_size_ = *client_batch_size_set.begin();
  if (TF_PREDICT_FALSE(client_batch_size_ < 0)) {
    return errors::InvalidArgument("Invalid client batch size ",
                                   client_batch_size_);
  }
  neff_batch_size_ = *neff_batch_size_set.begin();
  if (TF_PREDICT_FALSE(neff_batch_size_ < 0)) {
    return errors::InvalidArgument("Invalid NEFF batch size ",
                                   neff_batch_size_);
  }
  for (int idx = 0; idx < num_outputs; ++idx) {
    int batch_axis = info.output_batch_axis.i(idx);
    bool need_sharding = BatchAxisIsDynamic(batch_axis);
    outputs_need_sharding_.at(idx) = need_sharding;
    if (TF_PREDICT_TRUE(need_sharding)) {
      client_output_shapes_.at(idx).set_dim(batch_axis, client_batch_size_);
    }
  }
  VLOG(1) << "NeuronBatchSharder::Setup done after finding dynamic batch size";
  return Status::OK();
}

const std::vector<TensorShape>& NeuronBatchSharder::GetClientOutputShapes()
    const {
  return client_output_shapes_;
}

Status NeuronBatchSharder::ShardInputs(
    std::vector<std::vector<Tensor>>* sharded_inputs,
    const std::vector<Tensor>& inputs) {
  return ShardTensors(sharded_inputs, inputs, inputs_need_sharding_);
}

Status NeuronBatchSharder::ShardOutputs(
    std::vector<std::vector<Tensor>>* sharded_outputs,
    const std::vector<Tensor>& outputs) {
  return ShardTensors(sharded_outputs, outputs, outputs_need_sharding_);
}

Status NeuronBatchSharder::ShardTensors(
    std::vector<std::vector<Tensor>>* sharded_tensors,
    const std::vector<Tensor>& tensors,
    const std::vector<bool>& tensors_need_sharding) {
  if (TF_PREDICT_FALSE(can_skip_)) {
    return errors::InvalidArgument(__func__, " called on fixed-shape inputs");
  }
  int num_shards = (client_batch_size_ - 1) / neff_batch_size_ + 1;
  num_shards = std::max(num_shards, 1);
  sharded_tensors->reserve(num_shards);
  for (int sdidx = 0; sdidx < num_shards; ++sdidx) {
    sharded_tensors->emplace_back();
    std::vector<Tensor>& slices = sharded_tensors->back();
    slices.reserve(tensors.size());
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
      Tensor tensor = tensors.at(idx);
      if (tensors_need_sharding.at(idx)) {
        int dim0_start = sdidx * neff_batch_size_;
        int dim0_limit = dim0_start + neff_batch_size_;
        dim0_limit = std::min(dim0_limit, client_batch_size_);
        tensor = tensor.Slice(dim0_start, dim0_limit);
      }
      slices.push_back(tensor);
    }
  }
  return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow

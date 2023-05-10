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

#include "routine.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "core_range.h"
#include "dynamic_batch.h"
#include "executable.h"
#include "executable_info.h"
#include "host_memory.h"
#include "macros.h"
#include "placer.h"
#include "tensor_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace neuron {

NeuronRoutine::NeuronRoutine() : exe_(nullptr) {}

Status NeuronRoutine::MaybeInit(const NodeDef& node_def,
                                const std::string& session_handle) {
  tensorflow::mutex_lock lock(mu_);
  if (TF_PREDICT_TRUE(exe_ != nullptr)) {
    VLOG(1) << "NeuronRoutine is already initialized";
    return Status::OK();
  }
  NeuronCorePlacer& placer = NeuronCorePlacer::Singleton();
  TF_RETURN_IF_ERROR(placer.GetStatus());
  exe_ = absl::make_unique<NeuronDataParallelExecutable>();
  TF_RETURN_IF_ERROR(info_.ParseFromNodeDef(node_def));
  MaybeInitInputLocations();
  std::pair<Status, std::vector<NeuronCoreRange>> status_core_ranges =
      placer.GetParallelCoreRanges(info_, session_handle);
  TF_RETURN_IF_ERROR(status_core_ranges.first);
  // check to see if profiling is on
  if (const char* profile_dir_raw = std::getenv("NEURON_PROFILE")) {
    std::string profile_dir(profile_dir_raw);
    for (const auto& nc_range : status_core_ranges.second) {
      TF_RETURN_IF_ERROR(
          exe_->AddProfilingExecutable(info_, nc_range, profile_dir));
      break;
    }
  } else {
    for (const auto& nc_range : status_core_ranges.second) {
      VLOG(1) << "nc_start is " << nc_range.start_nc_;
      TF_RETURN_IF_ERROR(exe_->AddExecutable(info_.executable, nc_range));
    }
  }
  VLOG(1) << "NeuronRoutine::MaybeInit done";
  return Status::OK();
}

Status NeuronRoutine::MaybeShardedRun(NeuronBatchSharder* sharder,
                                      std::vector<Tensor>* inputs,
                                      std::vector<Tensor>* shuffle_buffers,
                                      std::vector<Tensor>* outputs) {
  if (sharder->CanSkip()) {
    // Fixed shape
    return RunWithIO(inputs, shuffle_buffers, outputs);
  }

  // Dynamic batch size
  std::vector<std::vector<Tensor>> sharded_inputs;
  std::vector<std::vector<Tensor>> sharded_buffers;
  std::vector<std::vector<Tensor>> sharded_outputs;
  TF_RETURN_IF_ERROR(sharder->ShardInputs(&sharded_inputs, *inputs));
  TF_RETURN_IF_ERROR(sharder->ShardInputs(&sharded_buffers, *shuffle_buffers));
  TF_RETURN_IF_ERROR(sharder->ShardOutputs(&sharded_outputs, *outputs));
  int64 num_shards = (int64)sharded_inputs.size();
  std::vector<Status> sharded_status;
  sharded_status.reserve(num_shards);
  for (int64 idx = 0; idx < num_shards; ++idx) {
    sharded_status.push_back(errors::Internal("Shard did not run"));
  }

  // Shard function
  auto RunShard = [&](int64 start, int64 limit) {
    for (int64 idx = start; idx < limit; ++idx) {
      std::vector<Tensor>* shard_inputs = &sharded_inputs.at(idx);
      std::vector<Tensor>* shard_buffers = &sharded_buffers.at(idx);
      std::vector<Tensor>* shard_outputs = &sharded_outputs.at(idx);
      Status status = RunWithIO(shard_inputs, shard_buffers, shard_outputs);
      sharded_status.at(idx) = status;
    }
  };

  // Run shards using CPU thread pool
  thread::ThreadPool* thread_pool =
      NeuronCorePlacer::Singleton().GetThreadPool();
  TFN_RETURN_IF_NULLPTR(thread_pool);
  int64 cost_per_unit = 1000000000;  // An arbitrary large number
  thread_pool->ParallelFor(num_shards, cost_per_unit, std::move(RunShard));
  for (const Status& status : sharded_status) {
    TF_RETURN_IF_ERROR(status);
  }
  return Status::OK();
}

void NeuronRoutine::MaybeInitInputLocations() {
  auto locs = info_.real_input_locations;
  auto real_names = info_.real_input_names;
  if (real_input_locations_.size() == 0) {
    if (real_names != nullptr && locs != nullptr) {
      real_input_locations_.reserve(locs->i_size());
      VLOG(1) << "Initialzing Real Input Locations, found " << locs->i_size()
              << " real inputs.";
      for (int idx = 0; idx < locs->i_size(); ++idx) {
        real_input_locations_.push_back(locs->i(idx));
      }
    }
  }
}

void NeuronRoutine::MaybeInitCache() {
  if (cache_.size() == 0) {
    // do this 64 times since max number of cores on a neuron device is 64
    for (int i = 0; i < 64; ++i) {
      cache_.push_back(std::vector<std::shared_ptr<NeuronDeviceBuffer>>());
    }
  }
}

int32_t NeuronRoutine::CoreIDToMemoryID(int32_t core_id, StringPiece& instance_type) {
  // A function for the --extract-weights feature that maps the models
  // to the location of the memory depending on instance type.
  // Note trn1 and inf2 instances always return core_id as memory cannot
  // be placed on a different core from the model.
  VLOG(1) << "calculating memory id for core_id " << core_id;
  VLOG(1) << "Detected extract weights model compiled for " << instance_type.data() << " instance";
  if (strcmp(instance_type.data(), "inf1.2xlarge") == 0 ||
      strcmp(instance_type.data(), "inf1.xlarge") == 0) {
    int32_t multiplier = 4 / exe_->GetNumLoadedModels();
    return core_id * multiplier;
  }
  else if (strcmp(instance_type.data(), "inf1.6xlarge") == 0) {
    int32_t multiplier = 16 / exe_->GetNumLoadedModels();
    return core_id * multiplier;
  }
  else if (strcmp(instance_type.data(), "inf1.32xlarge") == 0) {
    int32_t multiplier = 64 / exe_->GetNumLoadedModels();
    return core_id * multiplier;
  }
  else if (strcmp(instance_type.data(), "trn1.2xlarge")  == 0 ||
           strcmp(instance_type.data(), "trn1.32xlarge") == 0 ||
           strcmp(instance_type.data(), "inf2.xlarge")   == 0 ||
           strcmp(instance_type.data(), "inf2.8xlarge")  == 0 || 
           strcmp(instance_type.data(), "inf2.24xlarge") == 0 ||
           strcmp(instance_type.data(), "inf2.48xlarge") == 0 ) {

    return core_id;
  }
}

Status NeuronRoutine::MaybeShuffle(std::vector<Tensor>* inputs,
                                   std::vector<Tensor>* shuffle_buffers) {
  if (0 == info_.input_shuffles.tensor_size()) {
    return Status::OK();
  }
  for (int idx = 0; idx < info_.input_shuffles.tensor_size(); ++idx) {
    const TensorProto& shuffle = info_.input_shuffles.tensor(idx);
    if (TF_PREDICT_FALSE(!shuffle.int64_val_size())) {
      continue;
    }
    Tensor src = inputs->at(idx);
    Tensor* dst = &shuffle_buffers->at(idx);
    TF_RETURN_IF_ERROR(tensor_shuffle(dst, src, shuffle));
    inputs->at(idx) = *dst;
  }
  return Status::OK();
}

Status NeuronRoutine::RunWithIO(std::vector<Tensor>* inputs,
                                std::vector<Tensor>* shuffle_buffers,
                                std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(MaybeShuffle(inputs, shuffle_buffers));
  if (real_input_locations_.size() > 0) {
    int32_t core_id = exe_->GetRoundRobinId();
    int32_t memory_id = CoreIDToMemoryID(core_id, info_.instance_type);
    VLOG(1) << "Trying to create NeuronDeviceMemory on core " << memory_id
            << " for model on core " << core_id;
    NeuronDeviceMemory memory(memory_id);
    VLOG(1) << "Successfully created NeuronDeviceMemory on core " << memory_id
            << " for model on core " << core_id;
    MaybeInitCache();
    TF_RETURN_IF_ERROR(memory.SetupBuffers(
        info_, inputs, outputs, cache_.at(core_id), real_input_locations_));
    TF_RETURN_IF_ERROR(exe_->RunOnDeviceMemory(&memory, core_id));
    TF_RETURN_IF_ERROR(memory.CopyOutputBuffersToCPU(*outputs));
    return Status::OK();
  }
  NeuronHostMemory memory;
  TF_RETURN_IF_ERROR(memory.SetupBuffers(info_, inputs, outputs));
  TF_RETURN_IF_ERROR(memory.CopyCPUToInputBuffers(*inputs));
  TF_RETURN_IF_ERROR(exe_->RunOnHostMemory(&memory));
  TF_RETURN_IF_ERROR(memory.CopyOutputBuffersToCPU(*outputs));
  return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow

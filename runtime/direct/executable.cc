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

#include "executable.h"

#include <cstddef>

#include "../macros.h"
#include "absl/memory/memory.h"
#include "adaptor.h"
#include "core_range.h"
#include "host_memory.h"
#include "profiler_context.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace neuron {

NeuronExecutable::NeuronExecutable(StringPiece executable,
                                   const NeuronCoreRange& nc_range) {
  status_ =
      Nrt::Load(&rt_model_, executable, nc_range.start_nc_, nc_range.nc_count_);
}

NeuronExecutable::~NeuronExecutable() {
  if (status_.ok()) {
    Nrt::Unload(rt_model_);
  }
}

Status NeuronExecutable::RunOnHostMemory(NeuronHostMemory* memory) {
  TFN_RETURN_FAILED_PRECONDITION_IF_ERROR(status_);
  return Nrt::Execute(rt_model_, memory->input_buffer_map_.rt_buffer_map_,
                      &memory->output_buffer_map_.rt_buffer_map_);
}

Status NeuronExecutableProfiler::RunOnHostMemory(NeuronHostMemory* memory) {
  tensorflow::mutex_lock lock(mu_);
  TFN_RETURN_FAILED_PRECONDITION_IF_ERROR(status_);
  ProfilerContext StopsProfilingWhenItLeavesScope =
      ProfilerContext(rt_model_, profile_dir_, executable_);
  return Nrt::Execute(rt_model_, memory->input_buffer_map_.rt_buffer_map_,
                      &memory->output_buffer_map_.rt_buffer_map_);
}

NeuronExecutableProfiler::NeuronExecutableProfiler(
    StringPiece executable, const NeuronCoreRange& nc_range,
    std::string profile_dir)
    : NeuronExecutable::NeuronExecutable(executable, nc_range) {
  executable_ = executable;
  profile_dir_ = profile_dir;
}

Status NeuronDataParallelExecutable::AddExecutable(
    StringPiece executable, const NeuronCoreRange& nc_range) {
  tensorflow::mutex_lock lock(mu_);
  executables_.push_back(
      std::make_shared<NeuronExecutable>(executable, nc_range));

  Status status = executables_.back()->GetStatus();
  if (TF_PREDICT_FALSE(!status.ok())) {
    executables_.pop_back();
    return status;
  }
  return Status::OK();
}

Status NeuronDataParallelExecutable::AddProfilingExecutable(
    StringPiece executable, const NeuronCoreRange& nc_range,
    std::string profile_dir) {
  tensorflow::mutex_lock lock(mu_);
  executables_.push_back(std::make_shared<NeuronExecutableProfiler>(
      executable, nc_range, profile_dir));

  Status status = executables_.back()->GetStatus();
  if (TF_PREDICT_FALSE(!status.ok())) {
    executables_.pop_back();
    return status;
  }
  return Status::OK();
}

Status NeuronDataParallelExecutable::RunOnHostMemory(NeuronHostMemory* memory) {
  if (TF_PREDICT_FALSE(executables_.empty())) {
    return errors::FailedPrecondition(__func__, " called without executables");
  }
  return executables_.at(GetRoundRobinId())->RunOnHostMemory(memory);
}

size_t NeuronDataParallelExecutable::GetRoundRobinId() {
  tensorflow::mutex_lock lock(mu_);
  size_t exe_id = round_robin_exe_id_;
  ++round_robin_exe_id_;
  round_robin_exe_id_ %= executables_.size();
  return exe_id;
}

}  // namespace neuron
}  // namespace tensorflow

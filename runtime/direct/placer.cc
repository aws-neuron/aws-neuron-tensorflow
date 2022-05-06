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

#include "placer.h"
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>
#include "../env.h"
#include "adaptor.h"
#include "core_range.h"
#include "executable_info.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace neuron {

NeuronCorePlacer& NeuronCorePlacer::Singleton() {
  static NeuronCorePlacer placer;
  return placer;
}

NeuronCorePlacer::NeuronCorePlacer() {
  std::vector<std::pair<int, int>> engine_specs = parse_engine_specs();
  int num_cores_specified = 0;
  max_num_dup_ = MAX_NUM_CORES;
  std::string group_sizes = env_get("NEURONCORE_GROUP_SIZES", "");
  if (!group_sizes.empty()) {
    LOG(WARNING) << "NEURONCORE_GROUP_SIZES is being deprecated. "
                 << "Please see Neuron documentation for more details.";
  }
  specified_num_dup_.reserve(engine_specs.size());
  for (const auto& spec : engine_specs) {
    int num_cores = spec.first;
    bool num_cores_is_legal = 0 < num_cores && num_cores <= MAX_NUM_CORES;
    int num_dup = spec.second;
    bool num_dup_is_legal = 0 < num_dup && num_dup <= MAX_NUM_CORES;
    VLOG(1) << "num_cores=" << num_cores;
    VLOG(1) << "num_dup=" << num_dup;
    if (!(num_cores_is_legal && num_dup_is_legal)) {
      LOG(WARNING) << "NEURONCORE_GROUP_SIZES=" << group_sizes
                   << " looks ill-formatted -- ignoring.";
      break;
    }
    specified_num_dup_.push_back(num_dup);
    max_num_dup_ = std::min(max_num_dup_, num_dup);
    num_cores_specified += num_cores * num_dup;
  }
  VLOG(1) << "num_cores_specified=" << num_cores_specified;
  VLOG(1) << "max_num_dup_=" << max_num_dup_;
  if (max_num_dup_ > 1 && engine_specs.size() > 1) {
    status_ = errors::Unimplemented("NEURONCORE_GROUP_SIZES with ",
                                    engine_specs.size(),
                                    " groups and automatic data parallel (",
                                    max_num_dup_, ") is not implemented yet");
    return;
  }

  // Use limited number of cores if NEURONCORE_GROUP_SIZES is specified
  if (num_cores_specified) {
    std::string num_cores_specified_str = std::to_string(num_cores_specified);
    if (setenv("NEURON_RT_NUM_CORES",
               /*value=*/num_cores_specified_str.c_str(),
               /*overwrite=*/0)) {
      status_ = errors::InvalidArgument(
          "setenv failed; it is likely that "
          "NEURONCORE_GROUP_SIZES=",
          group_sizes, " is ill-formed");
      return;
    }
  }

  // Initialize runtime
  status_ = Nrt::Init();
  if (!status_.ok()) {
    return;
  }
  status_ = Nrt::GetCoreCount(&num_available_cores_);
  if (!status_.ok()) {
    Nrt::Close();
    return;
  }
  if (num_cores_specified > num_available_cores_) {
    Nrt::Close();
    status_ = errors::InvalidArgument(
        "NEURONCORE_GROUP_SIZES=", group_sizes, " requires ",
        num_cores_specified, " NeuronCores, but there are only ",
        num_available_cores_, " cores available");
    return;
  }
  core_pointer_ = 0;

  // Initialize ThreadPool
  thread_pool_ = absl::make_unique<thread::ThreadPool>(
      Env::Default(), ThreadOptions(), /*pool_name=*/"nrt_thread_pool",
      /*pool_size=*/num_available_cores_ * 2, /*low_latency_hint=*/false);
}

NeuronCorePlacer::~NeuronCorePlacer() {
  if (status_.ok()) {
    Nrt::Close();
  }
}

thread::ThreadPool* NeuronCorePlacer::GetThreadPool() {
  return thread_pool_.get();
}

std::pair<Status, std::vector<NeuronCoreRange>>
NeuronCorePlacer::GetParallelCoreRanges(const NeuronExecutableInfo& info,
                                        StringPiece session_handle) {
  tensorflow::mutex_lock lock(mu_);
  std::vector<NeuronCoreRange> core_ranges;
  if (TF_PREDICT_FALSE(!status_.ok())) {
    Status error = errors::FailedPrecondition(
        "NeuronCorePlacer::GetDataParallelCoreRange called without successful "
        "initialization");
    return std::make_pair(error, core_ranges);
  }
  int32_t nc_count = info.optimal_num_cores;
  if (TF_PREDICT_FALSE(nc_count > 1 || 1 == max_num_dup_)) {
    // Model parallel -- turn off data parallel
    std::pair<Status, NeuronCoreRange> status_core_range =
        UnsafeGetCoreRange(info, session_handle);
    core_ranges.push_back(status_core_range.second);
    return std::make_pair(status_core_range.first, core_ranges);
  }
  // Single-core executable -- place a copy on each core, up to max_num_dup_
  int32_t num_copies;

  bool automatic_multicore_enabled = (info.requested_num_cores != -1) ? true : false;
  
  if (automatic_multicore_enabled) {
    // Automatic Multicore
    num_copies = info.requested.num_cores;
  } else if (specified_num_dup_.empty()) {
    num_copies = std::min(info.max_num_duplicates, num_available_cores_);
  } else {
    // TODO: implement NEURONCORE_GROUP_SIZES + automatic data parallel
    num_copies = specified_num_dup_.at(0);
  }
  num_copies = std::min(num_copies, max_num_dup_);
  if (automatic_multicore_enabled) {
    for (int32_t start_nc = 0, start_nc < num_copies; ++start_nc) {
      core_ranges.emplace_back(start_nc, 1);
    }
  } else if (num_available_cores_ <= 4) {
    for (int32_t start_nc = 0; start_nc < num_copies; ++start_nc) {
      core_ranges.emplace_back(start_nc, nc_count);
    }
  } else {
    for (int32_t idx = 0; idx < num_copies; ++idx) {
      core_ranges.emplace_back(core_pointer_, nc_count);
      ++core_pointer_;
      core_pointer_ %= num_available_cores_;
    }
  }
  return std::make_pair(Status::OK(), core_ranges);
}

std::pair<Status, NeuronCoreRange> NeuronCorePlacer::UnsafeGetCoreRange(
    const NeuronExecutableInfo& info, StringPiece session_handle) {
  if (TF_PREDICT_FALSE(!status_.ok())) {
    Status error = errors::FailedPrecondition(
        "NeuronCorePlacer::GetCoreRange called without successful "
        "initialization");
    return std::make_pair(error, NeuronCoreRange(0, 0));
  }
  int32_t start_nc;
  int32_t nc_count = info.optimal_num_cores;
  start_nc = UnsafeGetNeuronCoreId(session_handle, nc_count);
  return std::make_pair(Status::OK(), NeuronCoreRange(start_nc, nc_count));
}

int32_t NeuronCorePlacer::UnsafeGetNeuronCoreId(StringPiece session_handle, int32_t nc_count) {
  int32_t core_id = core_pointer_;
  if (!session_handle.empty()) {
    if (sess_to_core_id_.count(session_handle)) {
      return sess_to_core_id_.at(session_handle);
    }
    sess_to_core_id_[session_handle] = core_id;
  }
  core_pointer_ += nc_count;
  core_pointer_ %= num_available_cores_;
  return core_id;
}

}  // namespace neuron
}  // namespace tensorflow

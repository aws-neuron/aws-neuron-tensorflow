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
#include "adaptor.h"
#include "core_range.h"
#include "executable_info.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace neuron {

NeuronCorePlacer& NeuronCorePlacer::Singleton() {
  static NeuronCorePlacer placer;
  return placer;
}

NeuronCorePlacer::NeuronCorePlacer() {
  status_ = Nrt::Init();
  if (!status_.ok()) {
    return;
  }
  status_ = Nrt::GetCoreCount(&num_available_cores_);
  if (!status_.ok()) {
    return;
  }
  core_pointer_ = 0;
}

NeuronCorePlacer::~NeuronCorePlacer() {
  if (status_.ok()) {
    Nrt::Close();
  }
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
  if (TF_PREDICT_FALSE(nc_count > 1)) {
    // Model parallel -- turn off data parallel
    std::pair<Status, NeuronCoreRange> status_core_range =
        UnsafeGetCoreRange(info, session_handle);
    core_ranges.push_back(status_core_range.second);
    return std::make_pair(status_core_range.first, core_ranges);
  }
  // Single-core executable -- place a copy on each core
  int32_t num_copies = std::min(info.max_num_duplicates, num_available_cores_);
  for (int32_t start_nc = 0; start_nc < num_copies; ++start_nc) {
    core_ranges.emplace_back(start_nc, nc_count);
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
  if (TF_PREDICT_FALSE(nc_count > 1)) {
    // Model parallel -- always load from core 0
    start_nc = 0;
  } else {
    // Single-core executable
    // Reuse core if session_handle is seen, or get a new core otherwise
    start_nc = UnsafeGetNeuronCoreId(session_handle);
  }
  return std::make_pair(Status::OK(), NeuronCoreRange(start_nc, nc_count));
}

int32_t NeuronCorePlacer::UnsafeGetNeuronCoreId(StringPiece session_handle) {
  if (sess_to_core_id_.count(session_handle)) {
    return sess_to_core_id_.at(session_handle);
  }
  int32_t core_id = core_pointer_;
  sess_to_core_id_.at(session_handle) = core_id;
  ++core_pointer_;
  core_pointer_ %= num_available_cores_;
  return core_id;
}

}  // namespace neuron
}  // namespace tensorflow

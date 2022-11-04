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

#ifndef TENSORFLOW_NEURON_RUNTIME_DIRECT_EXECUTABLE_H_
#define TENSORFLOW_NEURON_RUNTIME_DIRECT_EXECUTABLE_H_

#include <cstddef>
#include <memory>

#include "adaptor.h"
#include "core_range.h"
#include "executable_info.h"
#include "host_memory.h"
#include "macros.h"
#include "profiler_context.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace neuron {

class NeuronExecutable {
 public:
  NeuronExecutable(StringPiece executable, const NeuronCoreRange& nc_range);
  ~NeuronExecutable();
  Status GetStatus() { return status_; }
  virtual Status RunOnHostMemory(NeuronHostMemory* memory);
  virtual Status RunOnDeviceMemory(NeuronDeviceMemory* memory);

 protected:
  NrtModel rt_model_;
  Status status_;
  int32_t start_nc_;
  int32_t nc_count_;

 private:
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronExecutable);
};

class NeuronExecutableProfiler : public NeuronExecutable {
 public:
  NeuronExecutableProfiler(const NeuronExecutableInfo& info,
                           const NeuronCoreRange& nc_range,
                           const std::string& profile_dir);
  Status RunOnHostMemory(NeuronHostMemory* memory);

 private:
  tensorflow::mutex mu_;
  const NeuronExecutableInfo& info_;
  std::string profile_dir_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronExecutableProfiler);
};

class NeuronDataParallelExecutable {
 public:
  NeuronDataParallelExecutable() {}
  Status AddExecutable(StringPiece executable, const NeuronCoreRange& nc_range);
  Status AddProfilingExecutable(const NeuronExecutableInfo& info,
                                const NeuronCoreRange& nc_range,
                                const std::string& profile_dir);
  Status RunOnHostMemory(NeuronHostMemory* memory);
  Status RunOnDeviceMemory(NeuronDeviceMemory* memory, int32_t core_id);
  size_t GetRoundRobinId();
  size_t GetNumLoadedModels() { return executables_.size(); }

 private:
  tensorflow::mutex mu_;
  std::vector<std::shared_ptr<NeuronExecutable>> executables_;
  size_t round_robin_exe_id_ = 0;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronDataParallelExecutable);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DIRECT_EXECUTABLE_H_

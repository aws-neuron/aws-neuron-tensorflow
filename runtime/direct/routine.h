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

#ifndef TENSORFLOW_NEURON_RUNTIME_DIRECT_ROUTINE_H_
#define TENSORFLOW_NEURON_RUNTIME_DIRECT_ROUTINE_H_

#include <memory>
#include <string>
#include <vector>

#include "dynamic_batch.h"
#include "executable.h"
#include "executable_info.h"
#include "macros.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace neuron {

class NeuronRoutine {
 public:
  NeuronRoutine();
  Status MaybeInit(const NodeDef& node_def, const std::string& session_handle);
  const NeuronExecutableInfo& Info() { return info_; }
  Status MaybeShardedRun(NeuronBatchSharder* sharder,
                         std::vector<Tensor>* inputs,
                         std::vector<Tensor>* shuffle_buffers,
                         std::vector<Tensor>* outputs);

 private:
  Status MaybeShuffle(std::vector<Tensor>* inputs,
                      std::vector<Tensor>* shuffle_buffers);
  Status RunWithIO(std::vector<Tensor>* inputs,
                   std::vector<Tensor>* shuffle_buffers,
                   std::vector<Tensor>* outputs);
  void MaybeInitInputLocations();
  void MaybeInitCache();
  int32_t CoreIDToMemoryID(int32_t core_id, StringPiece& instance_type);
  tensorflow::mutex mu_;
  NeuronExecutableInfo info_;
  std::unique_ptr<NeuronDataParallelExecutable> exe_;
  std::vector<int> real_input_locations_;
  std::vector<std::vector<std::shared_ptr<NeuronDeviceBuffer>>> cache_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronRoutine);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DIRECT_ROUTINE_H_

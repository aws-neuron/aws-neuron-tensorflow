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

#ifndef TENSORFLOW_NEURON_RUNTIME_DIRECT_PLACER_H_
#define TENSORFLOW_NEURON_RUNTIME_DIRECT_PLACER_H_

#include <cstdint>
#include <unordered_map>
#include <utility>
#include "../macros.h"
#include "core_range.h"
#include "executable_info.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace neuron {

class NeuronCorePlacer {
 public:
  static NeuronCorePlacer& Singleton();
  Status GetStatus() { return status_; }
  std::pair<Status, std::vector<NeuronCoreRange>> GetParallelCoreRanges(
      const NeuronExecutableInfo& info, StringPiece session_handle);

 private:
  NeuronCorePlacer();
  ~NeuronCorePlacer();
  std::pair<Status, NeuronCoreRange> UnsafeGetCoreRange(
      const NeuronExecutableInfo& info, StringPiece session_handle);
  int32_t UnsafeGetNeuronCoreId(StringPiece session_handle);
  tensorflow::mutex mu_;
  Status status_;
  int32_t num_available_cores_;
  int32_t core_pointer_;
  int max_num_dup_;
  static const int MAX_NUM_CORES = 64;
  // A map from tensorflow session handles to NeuronCore IDs that is not going
  // to be cleaned up until the process dies. This is presumably OK as each new
  // session handle means a new tf 1.x Session or a new tf 2.x Function.
  std::unordered_map<StringPiece, int32_t, StringPieceHasher> sess_to_core_id_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronCorePlacer);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DIRECT_PLACER_H_

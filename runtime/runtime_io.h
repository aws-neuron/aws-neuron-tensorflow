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

#ifndef TENSORFLOW_NEURON_RUNTIME_RUNTIME_IO_H_
#define TENSORFLOW_NEURON_RUNTIME_RUNTIME_IO_H_

#include "shared_memory.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace neuron {

class ScopedRuntimeIO {
 public:
  ScopedRuntimeIO() {}
  Status copy_input_tensors(const std::vector<Tensor>& input_tensors,
                            std::vector<Tensor>* input_shm_tensors,
                            thread::ThreadPool* thread_pool);
  Status copy_input_tensors(const std::vector<Tensor>& input_tensors,
                            AttrList& input_shuffles,
                            std::vector<Tensor>* shuffle_buffers,
                            std::vector<Tensor>* input_shm_tensors);
  Status finish();
  ~ScopedRuntimeIO() {}
  RuntimeIO runtime_io_;

  TFN_DISALLOW_COPY_MOVE_ASSIGN(ScopedRuntimeIO);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_RUNTIME_IO_H_

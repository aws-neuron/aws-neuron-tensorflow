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
  Status setup(AttrList& input_names, const std::vector<Tensor>& input_tensors,
               AttrList& output_names,
               const std::vector<size_t>& output_tensor_sizes,
               const std::vector<Tensor*>& output_tensors, const uint32_t nn_id,
               thread::ThreadPool* thread_pool,
               std::shared_ptr<SharedMemoryAllocator> shm_alloc);
  Status copy_input_tensors(const std::vector<Tensor>& input_tensors,
                            std::vector<Tensor>* input_shm_tensors);
  Status copy_input_tensors(const std::vector<Tensor>& input_tensors,
                            AttrList& input_shuffles,
                            std::vector<Tensor>* shuffle_buffers,
                            std::vector<Tensor>* input_shm_tensors);
  std::vector<Tensor>* get_input_shm_tensors() { return &input_shm_tensors_; }
  Status finish();
  ~ScopedRuntimeIO() {}
  RuntimeIO runtime_io_;

 private:
  std::shared_ptr<SharedMemoryAllocator> shm_alloc_ = nullptr;
  std::vector<Tensor> input_shm_tensors_;
  std::vector<Tensor> output_shm_tensors_;
  thread::ThreadPool* thread_pool_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(ScopedRuntimeIO);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_RUNTIME_IO_H_

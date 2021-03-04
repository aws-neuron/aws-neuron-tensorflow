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

#include "tensorflow/core/lib/core/threadpool.h"
#include "shared_memory.h"

namespace tensorflow {
namespace neuron {

class ScopedRuntimeIO {
public:
    ScopedRuntimeIO() {}
    Status setup(AttrList &input_names,
                 const std::vector<const Tensor*> &input_tensors,
                 AttrList &output_names,
                 const std::vector<size_t> &output_tensor_sizes,
                 const std::vector<Tensor*> &output_tensors,
                 const uint32_t nn_id, thread::ThreadPool *thread_pool,
                 std::shared_ptr<SharedMemoryBufferManager> shm_mgr);
    Status copy_input_tensors(const std::vector<const Tensor*> &input_tensors);
    Status finish();
    ~ScopedRuntimeIO();
    RuntimeIO runtime_io_;
private:
    std::shared_ptr<SharedMemoryBufferManager> shm_mgr_ = nullptr;
    std::vector<Tensor> input_shm_tensors_;
    std::vector<SharedMemoryPtr> output_shm_bufs_;
    thread::ThreadPool *thread_pool_;
    TFN_DISALLOW_COPY_MOVE_ASSIGN(ScopedRuntimeIO);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_RUNTIME_IO_H_

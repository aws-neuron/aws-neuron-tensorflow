/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *     ==============================================================================*/

#ifndef TENSORFLOW_NEURON_KERNELS_NEURON_OP_H_
#define TENSORFLOW_NEURON_KERNELS_NEURON_OP_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/python/neuron/neuron_clib/neuron_clib.h"


namespace tensorflow {
namespace neuron {


class NeuronOp : public OpKernel {
public:
    explicit NeuronOp(OpKernelConstruction *ctx);
    void Compute(OpKernelContext *ctx) override;
    ~NeuronOp() override;

private:
    Status initialize();
    Status prepare_shared_memory();
    Status check_input_tensors(const std::vector<const Tensor*> &input_tensors);
    tensorflow::mutex mutex_model_;
    NeuronDevice *neuron_device_ = nullptr;
    uint32_t nn_id_ = NRT_INVALID_NN_ID;
    bool ready_ = false;
    bool unloaded_ = false;
    bool enable_dynamic_batch_size_ = false;
    std::vector<size_t> input_tensor_sizes_;
    uint32_t max_num_infers_ = 5;
    static const int64 INFER_SEM_MAX_CAPACITY = 2048;
    xla::Semaphore infer_sem_;
    bool infer_sem_initialized_ = false;
    std::unique_ptr<xla::Semaphore::ScopedReservation> infer_sem_reserve_ptr_;
    ProfilerInterface profile_;
    SharedMemoryManager shm_;
};


}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_KERNELS_NEURON_OP_H_

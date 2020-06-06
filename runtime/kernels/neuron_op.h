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
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/neuron/runtime/device.h"


namespace tensorflow {

constexpr char kNeuronOp[] = "NeuronOp";

namespace neuron {


class ScopedRuntimeIO {
public:
    ScopedRuntimeIO() {}
    Status setup(AttrList &input_names, const std::vector<const Tensor*> &input_tensors,
                 AttrList &output_names, const std::vector<Tensor*> &output_tensors,
                 const uint32_t nn_id, thread::ThreadPool *thread_pool,
                 SharedMemoryManager *shm_mgr) {
        shm_mgr_ = shm_mgr;
        if (nullptr != shm_mgr_) {
            shm_ = shm_mgr_->apply_for_shm();
        } else {
            shm_ = nullptr;
        }
        return runtime_io_.setup(
            input_names, input_tensors, output_names, output_tensors, nn_id, thread_pool, shm_);
    }
    Status finish() {
        return runtime_io_.finish();
    }
    ~ScopedRuntimeIO() {
        if (nullptr != shm_mgr_) {
            shm_mgr_->free_shm(shm_);
        }
    }
    ScopedRuntimeIO(const ScopedRuntimeIO &) = delete;
    ScopedRuntimeIO &operator=(const ScopedRuntimeIO &) = delete;
    ScopedRuntimeIO(ScopedRuntimeIO &&) = delete;
    ScopedRuntimeIO &operator=(ScopedRuntimeIO &&) = delete;
    RuntimeIO runtime_io_;
private:
    SharedMemoryManager *shm_mgr_ = nullptr;
    SharedMemory *shm_ = nullptr;
};


class NeuronOp : public OpKernel {
public:
    explicit NeuronOp(OpKernelConstruction *ctx);
    void Compute(OpKernelContext *ctx) override;
    ~NeuronOp() override;

private:
    Status initialize();
    Status prepare_shared_memory(const uint32_t max_num_infers);
    Status check_input_tensors(const std::vector<const Tensor*> &input_tensors);
    tensorflow::mutex mutex_model_;
    NeuronDevice *neuron_device_ = nullptr;
    uint32_t nn_id_ = NRT_INVALID_NN_ID;
    bool ready_ = false;
    std::vector<size_t> input_tensor_sizes_;
    uint32_t max_num_infers_ = 5;
    static const int64 INFER_SEM_MAX_CAPACITY = 2048;
    xla::Semaphore infer_sem_;
    bool infer_sem_initialized_ = false;
    std::unique_ptr<xla::Semaphore::ScopedReservation> infer_sem_reserve_ptr_;
    ProfilerInterface profile_;
    SharedMemoryManager *shm_mgr_ = nullptr;
    uint64 last_infer_timestamp_ = 0;
    static const uint64 INFER_NEED_PING_MICROSEC_ = 1024 * 1024;
};


}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_KERNELS_NEURON_OP_H_

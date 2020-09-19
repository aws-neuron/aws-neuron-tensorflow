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

#ifndef TENSORFLOW_NEURON_RUNTIME_KERNELS_NEURON_OP_H_
#define TENSORFLOW_NEURON_RUNTIME_KERNELS_NEURON_OP_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/neuron/runtime/macros.h"
#include "tensorflow/neuron/runtime/shared_memory_io.h"
#include "tensorflow/neuron/runtime/shared_memory.h"
#include "tensorflow/neuron/runtime/device.h"


namespace tensorflow {
namespace neuron {


class ScopedRuntimeIO {
public:
    ScopedRuntimeIO() {}
    Status setup(AttrList &input_names,
                 const std::vector<size_t> &input_tensor_sizes,
                 const std::vector<const Tensor*> &input_tensors,
                 AttrList &output_names,
                 const std::vector<size_t> &output_tensor_sizes,
                 const std::vector<Tensor*> &output_tensors,
                 const uint32_t nn_id, thread::ThreadPool *thread_pool,
                 std::shared_ptr<SharedMemoryBufferManager> shm_mgr) {
        shm_mgr_ = shm_mgr;
        SharedMemory scoped_shm;
        SharedMemory *shm = nullptr;
        if (nullptr != shm_mgr_ && shm_mgr_->is_valid()) {
            bool allocation_ok = true;
            for (size_t buf_size : input_tensor_sizes) {
                SharedMemoryPtr shm_buf = shm_mgr_->allocate_shm(buf_size);
                if (nullptr == shm_buf) {
                    allocation_ok = false;
                    break;
                }
                input_shm_bufs_.push_back(shm_buf);
            }
            for (size_t buf_size : output_tensor_sizes) {
                SharedMemoryPtr shm_buf = shm_mgr_->allocate_shm(buf_size);
                if (nullptr == shm_buf) {
                    allocation_ok = false;
                    break;
                }
                output_shm_bufs_.push_back(shm_buf);
            }
            if (allocation_ok) {
                for (auto shm_buf : input_shm_bufs_) {
                    scoped_shm.input_paths_.push_back(shm_buf->get_path());
                    scoped_shm.input_ptrs_.push_back(shm_buf->get_ptr());
                }
                for (auto shm_buf : output_shm_bufs_) {
                    scoped_shm.output_paths_.push_back(shm_buf->get_path());
                    scoped_shm.output_ptrs_.push_back(shm_buf->get_ptr());
                }
                shm = &scoped_shm;
            }
        }
        return runtime_io_.setup(
            input_names, input_tensors, output_names, output_tensors, nn_id, thread_pool, shm);
    }
    Status finish() {
        return runtime_io_.finish();
    }
    ~ScopedRuntimeIO() {
        if (nullptr != shm_mgr_) {
            for (auto shm_buf : input_shm_bufs_) {
                shm_mgr_->free_shm(shm_buf);
            }
            for (auto shm_buf : output_shm_bufs_) {
                shm_mgr_->free_shm(shm_buf);
            }
        }
    }
    ScopedRuntimeIO(const ScopedRuntimeIO &) = delete;
    ScopedRuntimeIO &operator=(const ScopedRuntimeIO &) = delete;
    ScopedRuntimeIO(ScopedRuntimeIO &&) = delete;
    ScopedRuntimeIO &operator=(ScopedRuntimeIO &&) = delete;
    RuntimeIO runtime_io_;
private:
    std::shared_ptr<SharedMemoryBufferManager> shm_mgr_ = nullptr;
    std::vector<SharedMemoryPtr> input_shm_bufs_;
    std::vector<SharedMemoryPtr> output_shm_bufs_;
};


class NeuronOp : public OpKernel {
public:
    explicit NeuronOp(OpKernelConstruction *ctx);
    void Compute(OpKernelContext *ctx) override;
    ~NeuronOp() override;

private:
    Status initialize();
    Status check_input_tensors(const std::vector<const Tensor*> &input_tensors);
    tensorflow::mutex mutex_model_;
    NeuronDevice *neuron_device_ = nullptr;
    uint32_t nn_id_ = NRT_INVALID_NN_ID;
    bool ready_ = false;
    std::vector<size_t> input_tensor_sizes_;
    std::vector<size_t> output_tensor_sizes_;
    uint32_t max_num_infers_ = 5;
    static const int64 INFER_SEM_MAX_CAPACITY = 2048;
    xla::Semaphore infer_sem_;
    bool infer_sem_initialized_ = false;
    std::shared_ptr<xla::Semaphore::ScopedReservation> infer_sem_reserve_ptr_;
    ProfilerInterface profile_;
    uint64 last_infer_timestamp_ = 0;
    static const uint64 INFER_NEED_PING_MICROSEC_ = 1024 * 1024;
};


}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_KERNELS_NEURON_OP_H_

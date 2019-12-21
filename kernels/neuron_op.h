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
#include "tensorflow/compiler/xla/python/semaphore.h"
#include "tensorflow/python/neuron/neuron_clib/neuron_clib.h"


namespace tensorflow {
namespace neuron {


typedef const AttrValue_ListValue AttrList;


class NeuronOp : public OpKernel {
public:
    explicit NeuronOp(OpKernelConstruction *ctx);
    void Compute(OpKernelContext *ctx) override;
    ~NeuronOp() override;

private:
    Status initialize();
    Status load(const AttrList &model_config);
    Status prepare_shared_memory();
    Status start_model();
    void profile_dump_info();
    void profile_start_session();
    void profile_stop_session();
    Status infer(std::vector<Tensor*> *output_tensors,
                 const std::vector<const Tensor*> &input_tensors,
                 FALTimestamps *timestamps);
    Status infer_post(uint64_t *cookie,
                      const std::vector<const Tensor*> &input_tensors);
    Status infer_wait(nrt::infer_response *response, uint64_t cookie);
    Status copy_output_tensors(std::vector<Tensor*> *output_tensors,
                               const nrt::infer_response &response);
    tensorflow::mutex mutex_model_;
    NeuronDevice *neuron_device_ = nullptr;
    std::string nrtd_address_;
    std::unique_ptr<nrt::nmgr_v1::Stub> stub_;
    uint32_t nn_id_ = NRT_INVALID_NN_ID;
    bool load_done_ = false;
    bool use_shared_memory_ = false;
    bool ready_ = false;
    bool unloaded_ = false;
    bool enable_dynamic_batch_size_ = false;
    std::vector<SharedMemory> input_shms_;
    std::vector<SharedMemory> output_shms_;
    std::vector<size_t> input_tensor_sizes_;
    uint32_t max_num_infers_;
    static const int64 DEFAULT_MAX_NUM_INFER = 4;
    static const int64 NRTD_INSUFFICIENT_NUM_INFER = 1;
    static const int64 NRTD_NUM_CPU_THREADS = 3;
    static const int64 INFER_SEM_MAX_CAPACITY = 2048;
    static const int64 HARD_MAX_NUM_THREADS = 1024;
    xla::Semaphore infer_sem_;
    bool infer_sem_initialized_ = false;
    std::unique_ptr<xla::Semaphore::ScopedReservation> infer_sem_reserve_ptr_;
    int profile_session_id_ = 0;
    bool profile_enabled_ = false;
    std::string profile_dir_ = "";
    std::string profile_session_filename_ = "";
};


}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_KERNELS_NEURON_OP_H_

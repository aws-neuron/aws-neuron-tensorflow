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

#ifndef TENSORFLOW_CONTRIB_KAENA_KERNELS_INFERENTIA_OP_H_
#define TENSORFLOW_CONTRIB_KAENA_KERNELS_INFERENTIA_OP_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/python/neuron/neuron_clib/neuron_clib.h"


namespace tensorflow {
namespace kaena {


class NeuronOp : public OpKernel {
public:
    explicit NeuronOp(OpKernelConstruction *ctx);
    void Compute(OpKernelContext *ctx) override;
    ~NeuronOp() override;

private:
    Status initialize();
    Status prepare_shared_memory();
    Status start_model();
    void profile_dump_info();
    void profile_start_session();
    void profile_stop_session();
    Status infer(std::vector<Tensor*> *output_tensors,
                 const std::vector<const Tensor*> &input_tensors,
                 FALTimestamps *timestamps);
    Status infer_post(uint64_t *infer_post_cookie,
                      const std::vector<const Tensor*> &input_tensors);
    Status infer_wait(std::vector<Tensor*> *output_tensors,
                      uint64_t infer_post_cookie);
    tensorflow::mutex load_mutex_;
    NeuronDevice *neuron_device_ = nullptr;
    std::string krtd_server_;
    std::unique_ptr<nrt::nmgr_v1::Stub> stub_;
    uint32_t krt_nn_id_ = NRT_INVALID_NN_ID;
    bool krt_load_done_ = false;
    bool use_shared_memory_ = false;
    bool ready_ = false;
    std::vector<SharedMemory> input_shms_;
    std::vector<SharedMemory> output_shms_;
    std::vector<SharedMemoryAllocator> output_shm_allocs_;
    std::vector<size_t> input_tensor_sizes_;
    std::vector<Tensor> output_tensors_;
    uint32_t infer_timeout_;
    uint32_t infer_queue_length_;
    int profile_session_id_ = 0;
    bool profile_enabled_ = false;
    std::string profile_dir_ = "";
    std::string profile_session_filename_ = "";
};


}  // namespace kaena
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_KAENA_KERNELS_INFERENTIA_OP_H_

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


class InferentiaOp : public OpKernel {
public:
    explicit InferentiaOp(OpKernelConstruction *context);
    void Compute(OpKernelContext *context) override;
    ~InferentiaOp() override;

private:
    tensorflow::Status initialize(const std::string &executable,
                                  const std::vector<std::string> &input_names,
                                  const std::vector<std::string> &output_names,
                                  const std::vector<DataType> &output_dtypes,
                                  const std::vector<TensorShape> &output_shapes);
    tensorflow::Status prepare_shared_memory(
        const std::vector<DataType> &output_dtypes,
        const std::vector<TensorShape> &output_shapes);
    tensorflow::Status start_model();
    void profile_dump_info(const std::string &graph_def, const std::string &executable);
    tensorflow::Status profile_start_session();
    void profile_stop_session();
    tensorflow::Status infer(const std::vector<const Tensor*> &input_tensors,
                             FALTimestamps *timestamps);
    tensorflow::Status infer_post(uint64_t *infer_post_cookie,
                                  const std::vector<const Tensor*> &input_tensors);
    tensorflow::Status infer_wait(uint64_t infer_post_cookie);
    std::string op_name_;
    TPBGroup *tpb_group_ = nullptr;
    std::string krtd_server_;
    std::unique_ptr<krt::kmgr_v1::Stub> stub_;
    uint32_t krt_nn_id_ = KRT_INVALID_NN_ID;
    bool krt_load_done_ = false;
    bool use_shared_memory_ = false;
    bool ready_ = false;
    std::vector<SharedMemory> input_shms_;
    std::vector<SharedMemory> output_shms_;
    std::vector<SharedMemoryAllocator> output_shm_allocs_;
    std::vector<int> input_batch_axis_;
    std::vector<int> output_batch_axis_;
    std::vector<std::string> input_names_;
    std::vector<DataType> input_dtypes_;
    std::vector<TensorShape> input_shapes_;
    std::vector<size_t> input_tensor_sizes_;
    std::vector<std::string> output_names_;
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

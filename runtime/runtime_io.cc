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

#include "runtime_io.h"
#include "tensor_util.h"

namespace tensorflow {
namespace neuron {

Status ScopedRuntimeIO::copy_input_tensors(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>* input_shm_tensors,
    thread::ThreadPool* thread_pool) {
  if (TF_PREDICT_TRUE(runtime_io_.use_shm())) {
    CHECK_VALID_PTR(input_shm_tensors);
    CHECK_SIZES_MATCH(input_shm_tensors->size(), input_tensors.size());
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      Tensor* dst = &input_shm_tensors->at(idx);
      TF_RETURN_IF_ERROR(tensor_copy(dst, input_tensors.at(idx), thread_pool));
    }
  }
  return runtime_io_.copy_input_tensors(input_tensors);
}

Status ScopedRuntimeIO::copy_input_tensors(
    const std::vector<Tensor>& input_tensors, AttrList& input_shuffles,
    std::vector<Tensor>* shuffle_buffers,
    std::vector<Tensor>* input_shm_tensors) {
  uint64 start_timestamp = Env::Default()->NowMicros();
  CHECK_SIZES_MATCH(input_shuffles.tensor_size(), input_tensors.size());
  if (TF_PREDICT_TRUE(runtime_io_.use_shm())) {
    CHECK_VALID_PTR(input_shm_tensors);
    CHECK_SIZES_MATCH(input_shm_tensors->size(), input_tensors.size());
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      Tensor* dst = &input_shm_tensors->at(idx);
      const TensorProto& shuffle = input_shuffles.tensor(idx);
      TF_RETURN_IF_ERROR(tensor_shuffle(dst, input_tensors.at(idx), shuffle));
    }
    TF_RETURN_IF_ERROR(runtime_io_.copy_input_tensors(input_tensors));
  } else {
    if (TF_PREDICT_FALSE(nullptr == shuffle_buffers)) {
      return errors::Internal(
          "need allocated input shuffle buffers for non-shared-memory");
    }
    if (TF_PREDICT_FALSE(shuffle_buffers->size() != input_tensors.size())) {
      return errors::Internal(
          "wrong number of allocated input shuffle buffers");
    }
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      Tensor* dst = &shuffle_buffers->at(idx);
      const TensorProto& shuffle = input_shuffles.tensor(idx);
      TF_RETURN_IF_ERROR(tensor_shuffle(dst, input_tensors.at(idx), shuffle));
    }
    TF_RETURN_IF_ERROR(runtime_io_.copy_input_tensors(*shuffle_buffers));
  }
  uint64 elapsed = Env::Default()->NowMicros() - start_timestamp;
  VLOG(1) << "input copy and shuffle for " << input_tensors.size()
          << " tensors took " << elapsed << " us";
  return Status::OK();
}

Status ScopedRuntimeIO::finish() { return runtime_io_.finish(); }

}  // namespace neuron
}  // namespace tensorflow

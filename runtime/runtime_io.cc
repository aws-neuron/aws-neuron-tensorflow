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

Status ScopedRuntimeIO::setup(
    AttrList& input_names, const std::vector<const Tensor*>& input_tensors,
    AttrList& output_names, const std::vector<size_t>& output_tensor_sizes,
    const std::vector<Tensor*>& output_tensors, const uint32_t nn_id,
    thread::ThreadPool* thread_pool,
    std::shared_ptr<SharedMemoryAllocator> shm_alloc) {
  shm_alloc_ = shm_alloc;
  thread_pool_ = thread_pool;
  bool use_shm = false;
  std::vector<StringPiece> input_paths;
  std::vector<StringPiece> output_paths;
  if (nullptr != shm_alloc_ && shm_alloc_->is_valid()) {
    input_shm_tensors_.reserve(input_tensors.size());
    std::vector<SharedMemoryPtr> input_shm_bufs;
    std::vector<SharedMemoryPtr> output_shm_bufs;
    for (const Tensor* tensor : input_tensors) {
      TensorShape shape = tensor->shape();
      DataType dtype = tensor->dtype();
      AllocationAttributes attr;
      input_shm_tensors_.emplace_back(shm_alloc_.get(), dtype, shape, attr);
      const Tensor& shm_tensor = input_shm_tensors_.back();
      SharedMemoryPtr shm_buf = shm_alloc_->get_shm_ptr(shm_tensor);
      input_shm_bufs.push_back(shm_buf);
    }
    for (size_t buf_size : output_tensor_sizes) {
      TensorShape shape({buf_size});
      DataType dtype(DT_UINT8);
      AllocationAttributes attr;
      output_shm_tensors_.emplace_back(shm_alloc_.get(), dtype, shape, attr);
      const Tensor& shm_tensor = output_shm_tensors_.back();
      SharedMemoryPtr shm_buf = shm_alloc_->get_shm_ptr(shm_tensor);
      output_shm_bufs.push_back(shm_buf);
    }
    for (auto shm_buf : input_shm_bufs) {
      input_paths.push_back(shm_buf->get_path());
    }
    for (auto shm_buf : output_shm_bufs) {
      output_paths.push_back(shm_buf->get_path());
    }
    use_shm = true;
  } else {
    for (size_t buf_size : output_tensor_sizes) {
      TensorShape shape({buf_size});
      DataType dtype(DT_UINT8);
      output_shm_tensors_.emplace_back(dtype, shape);
    }
  }
  std::vector<Tensor*> output_shm_tensors;
  output_shm_tensors.reserve(output_shm_tensors_.size());
  for (auto& tensor : output_shm_tensors_) {
    output_shm_tensors.push_back(&tensor);
  }
  return runtime_io_.setup(input_names, output_names, output_tensors,
                           output_shm_tensors, nn_id, use_shm, input_paths,
                           output_paths, thread_pool);
}

Status ScopedRuntimeIO::copy_input_tensors(
    const std::vector<const Tensor*>& input_tensors,
    std::vector<Tensor>* input_shm_tensors) {
  if (TF_PREDICT_TRUE(runtime_io_.use_shm())) {
    CHECK_VALID_PTR(input_shm_tensors);
    CHECK_SIZES_MATCH(input_shm_tensors->size(), input_tensors.size());
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      Tensor* dst = &input_shm_tensors->at(idx);
      TF_RETURN_IF_ERROR(tensor_copy(dst, *input_tensors[idx], thread_pool_));
    }
  }
  return runtime_io_.copy_input_tensors(input_tensors);
}

Status ScopedRuntimeIO::copy_input_tensors(
    const std::vector<const Tensor*>& input_tensors, AttrList& input_shuffles,
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
      TF_RETURN_IF_ERROR(tensor_shuffle(dst, *input_tensors[idx], shuffle));
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
    std::vector<const Tensor*> shuffled_input_tensors(input_tensors.size());
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      Tensor* dst = &shuffle_buffers->at(idx);
      const TensorProto& shuffle = input_shuffles.tensor(idx);
      TF_RETURN_IF_ERROR(tensor_shuffle(dst, *input_tensors[idx], shuffle));
      shuffled_input_tensors[idx] = dst;
    }
    TF_RETURN_IF_ERROR(runtime_io_.copy_input_tensors(shuffled_input_tensors));
  }
  uint64 elapsed = Env::Default()->NowMicros() - start_timestamp;
  VLOG(1) << "input copy and shuffle for " << input_tensors.size()
          << " tensors took " << elapsed << " us";
  return Status::OK();
}

Status ScopedRuntimeIO::finish() { return runtime_io_.finish(); }

}  // namespace neuron
}  // namespace tensorflow

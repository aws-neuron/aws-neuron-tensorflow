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

#include "tensor_util.h"
#include "shared_memory_io.h"
#include "runtime_io.h"

namespace tensorflow {
namespace neuron {

Status ScopedRuntimeIO::setup(AttrList &input_names,
                              const std::vector<const Tensor*> &input_tensors,
                              AttrList &output_names,
                              const std::vector<size_t> &output_tensor_sizes,
                              const std::vector<Tensor*> &output_tensors,
                              const uint32_t nn_id, thread::ThreadPool *thread_pool,
                              std::shared_ptr<SharedMemoryBufferManager> shm_mgr) {
    shm_mgr_ = shm_mgr;
    thread_pool_ = thread_pool;
    SharedMemory scoped_shm;
    SharedMemory *shm = nullptr;
    if (nullptr != shm_mgr_ && shm_mgr_->is_valid()) {
        bool allocation_ok = true;
        input_shm_tensors_.reserve(input_tensors.size());
        std::vector<SharedMemoryPtr> input_shm_bufs;
        for (const Tensor *tensor : input_tensors) {
            TensorShape shape = tensor->shape();
            DataType dtype = tensor->dtype();
            AllocationAttributes attr;
            input_shm_tensors_.emplace_back(shm_mgr_.get(), dtype, shape, attr);
            const void *temp_ptr = input_shm_tensors_.back().tensor_data().data();
            SharedMemoryPtr shm_buf = shm_mgr_->get_shm_ptr_from_ptr(temp_ptr);
            if (nullptr == shm_buf) {
                allocation_ok = false;
                break;
            }
            input_shm_bufs.push_back(shm_buf);
        }
        for (size_t buf_size : output_tensor_sizes) {
            SharedMemoryPtr shm_buf = shm_mgr_->allocate_shm(1, buf_size);
            if (nullptr == shm_buf) {
                allocation_ok = false;
                break;
            }
            output_shm_bufs_.push_back(shm_buf);
        }
        if (allocation_ok) {
            for (auto shm_buf : input_shm_bufs) {
                scoped_shm.input_paths_.push_back(shm_buf->get_path());
            }
            for (auto shm_buf : output_shm_bufs_) {
                scoped_shm.output_paths_.push_back(shm_buf->get_path());
                scoped_shm.output_ptrs_.push_back(shm_buf->get_ptr());
            }
            shm = &scoped_shm;
        }
    }
    return runtime_io_.setup(input_names, output_names, output_tensors, nn_id, thread_pool, shm);
}

Status ScopedRuntimeIO::copy_input_tensors(const std::vector<const Tensor*> &input_tensors) {
    if (TF_PREDICT_TRUE(runtime_io_.use_shm())) {
        if (TF_PREDICT_FALSE(input_shm_tensors_.size() != input_tensors.size())) {
            return errors::InvalidArgument(
                "size mismatch between input_shm_tensors_ and input_tensors");
        }
        for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
            StringPiece tensor_data(input_tensors[idx]->tensor_data());
            TF_RETURN_IF_ERROR(tensor_memcpy(thread_pool_, &input_shm_tensors_[idx], tensor_data));
        }
    }
    return runtime_io_.copy_input_tensors(input_tensors);
}

Status ScopedRuntimeIO::copy_input_tensors(const std::vector<const Tensor*> &input_tensors,
                                           AttrList &input_shuffles,
                                           std::vector<Tensor> *shuffle_buffers,
                                           std::vector<Tensor*> *input_shm_tensors) {
    uint64 start_timestamp = Env::Default()->NowMicros();
    std::vector<Tensor*> input_shm_ptrs;
    if (nullptr == input_shm_tensors) {
        for (Tensor &shm_tensor : input_shm_tensors_) {
            input_shm_ptrs.push_back(&shm_tensor);
        }
        input_shm_tensors = &input_shm_ptrs;
    }
    int64 num_shuffles = input_shuffles.tensor_size();
    int64 num_tensors = input_tensors.size();
    if (TF_PREDICT_FALSE(num_shuffles != num_tensors)) {
        return errors::InvalidArgument(num_shuffles, " shuffles and ", num_tensors, " tensors ");
    }
    if (TF_PREDICT_TRUE(runtime_io_.use_shm())) {
        if (TF_PREDICT_FALSE(input_shm_tensors->size() != input_tensors.size())) {
            return errors::InvalidArgument(
                "size mismatch between input_shm_tensors and input_tensors");
        }
        for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
            Tensor *dst = input_shm_tensors->at(idx);
            const TensorProto &shuffle = input_shuffles.tensor(idx);
            TF_RETURN_IF_ERROR(tensor_shuffle(dst, *input_tensors[idx], shuffle));
        }
        TF_RETURN_IF_ERROR(runtime_io_.copy_input_tensors(input_tensors));
    } else {
        if (TF_PREDICT_FALSE(nullptr == shuffle_buffers)) {
            return errors::Internal("need allocated input shuffle buffers for non-shared-memory");
        }
        if (TF_PREDICT_FALSE(shuffle_buffers->size() != input_tensors.size())) {
            return errors::Internal("wrong number of allocated input shuffle buffers");
        }
        std::vector<const Tensor*> shuffled_input_tensors(input_tensors.size());
        for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
            Tensor *dst = &shuffle_buffers->at(idx);
            const TensorProto &shuffle = input_shuffles.tensor(idx);
            TF_RETURN_IF_ERROR(tensor_shuffle(dst, *input_tensors[idx], shuffle));
            shuffled_input_tensors[idx] = dst;
        }
        TF_RETURN_IF_ERROR(runtime_io_.copy_input_tensors(shuffled_input_tensors));
    }
    uint64 elapsed = Env::Default()->NowMicros() - start_timestamp;
    VLOG(1) << "input copy and shuffle for " << num_tensors << " tensors took " << elapsed << " us";
    return Status::OK();
}

Status ScopedRuntimeIO::finish() {
    return runtime_io_.finish();
}

ScopedRuntimeIO::~ScopedRuntimeIO() {
    if (nullptr != shm_mgr_) {
        for (auto shm_buf : output_shm_bufs_) {
            shm_mgr_->free_shm(shm_buf);
        }
    }
}

}  // namespace neuron
}  // namespace tensorflow

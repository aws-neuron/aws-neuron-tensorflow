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

#ifndef TENSORFLOW_NEURON_RUNTIME_SHARED_MEMORY_H_
#define TENSORFLOW_NEURON_RUNTIME_SHARED_MEMORY_H_

#include "macros.h"
#include "runtime_grpc.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace neuron {

class SharedMemoryBuffer {
 public:
  SharedMemoryBuffer(const size_t id, const uint64_t session_id,
                     const size_t alignment, const size_t size,
                     std::shared_ptr<RuntimeGRPC> runtime);
  ~SharedMemoryBuffer();
  // path_ is assigned when rtd shm_map has returned success
  bool is_valid() { return !path_.empty(); }
  size_t get_id() { return id_; }
  void* get_ptr() { return ptr_; }
  size_t get_size() { return size_; }
  StringPiece get_path() { return StringPiece(path_); }
  std::string debug_string();

 private:
  const size_t id_;
  std::shared_ptr<RuntimeGRPC> runtime_ = nullptr;
  void* ptr_ = nullptr;
  void* physical_ptr_ = nullptr;
  size_t size_ = 0;
  size_t physical_size_ = 0;
  std::string path_ = "";
  TFN_DISALLOW_COPY_MOVE_ASSIGN(SharedMemoryBuffer);
};

typedef std::shared_ptr<SharedMemoryBuffer> SharedMemoryPtr;

class SharedMemoryAllocator : public Allocator {
 public:
  SharedMemoryAllocator();
  ~SharedMemoryAllocator() override {}
  Status initialize(const uint64_t session_id, const std::string& nrtd_address);
  bool is_valid() { return is_valid_; }
  std::string Name() override { return "AwsNeuronSharedMemory"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  size_t AllocatedSizeSlow(const void* ptr) const override;
  bool is_shm_tensor(const Tensor& tensor);
  SharedMemoryPtr get_shm_ptr(const Tensor& tensor);

 private:
  SharedMemoryPtr allocate_shm(const size_t alignment, const size_t size);
  void free_shm_unsafe(SharedMemoryPtr shm);
  tensorflow::mutex mutex_;
  uint64_t session_id_ = RuntimeSession::INVALID_ID;
  std::shared_ptr<RuntimeGRPC> runtime_ = nullptr;
  bool is_valid_ = false;
  std::vector<SharedMemoryPtr> buffer_vec_;
  std::unordered_map<size_t, std::unordered_set<size_t> >
      size_to_free_buffer_id_;
  std::unordered_map<const void*, size_t> ptr_to_id_;
  std::atomic<int> single_allocation_warning_count_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(SharedMemoryAllocator);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_SHARED_MEMORY_H_

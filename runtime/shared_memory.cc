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

#include "shared_memory.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "env.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace neuron {

class ShmFile {
 public:
  ShmFile(const std::string& name) {
    name_ = name;
    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
    shm_open_fd_ = ::shm_open(name.c_str(), O_CREAT | O_RDWR, mode);
    if (shm_open_fd_ >= 0) {
      if (::fchmod(shm_open_fd_, mode) < 0) {
        shm_open_fd_ = -1;
      } else {
        shm_fd_ = shm_open_fd_;
      }
    }
  }
  ~ShmFile() {
    if (shm_open_fd_ >= 0) {
      ::close(shm_open_fd_);
      SYS_FAIL_LOG(shm_unlink(name_.c_str()) < 0, "shm_unlink");
    }
  }
  int shm_fd_ = -1;

 private:
  int shm_open_fd_ = -1;
  std::string name_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(ShmFile);
};

static std::string gen_shm_path() {
  std::string filename = "/aws_neuron_runtime_";
  for (size_t i = 0; i < 64; ++i) {
    if (Env::Default()->CreateUniqueFileName(&filename, "")) {
      return filename;
    }
    Env::Default()->SleepForMicroseconds(1);
  }
  return "";
}

SharedMemoryBuffer::SharedMemoryBuffer(const size_t id,
                                       const uint64_t session_id,
                                       const size_t alignment,
                                       const size_t size,
                                       std::shared_ptr<RuntimeGRPC> runtime)
    : id_(id) {
  VLOG(1) << "entering SharedMemoryBuffer constructor";
  if (nullptr == runtime) {
    LOG(ERROR) << "runtime is not initialized";
    return;
  }
  bool alignment_is_power_of_two =
      (alignment != 0) && (alignment & (alignment - 1)) == 0;
  if (!alignment_is_power_of_two) {
    LOG(ERROR) << "alignment is not power of 2";
    return;
  }
  runtime_ = runtime;
  std::string path = gen_shm_path();
  if (path.empty()) {
    LOG(ERROR) << "cannot generate unique file name for shared memory";
    return;
  }
  size_ = size;
  physical_size_ = size;
  size_t page_size = ::getpagesize();
  if (alignment > page_size) {
    physical_size_ += alignment;
  } else {
    VLOG(1) << "no need for padding as alignment requirement " << alignment
            << " is less than page size " << page_size;
  }
  ShmFile shm_file(path);
  SYS_FAIL_LOG_RETURN(shm_file.shm_fd_ < 0, "shm_open");
  SYS_FAIL_LOG_RETURN(::ftruncate(shm_file.shm_fd_, physical_size_) < 0,
                      "ftruncate");
  physical_ptr_ =
      ::mmap(NULL, physical_size_, PROT_WRITE, MAP_SHARED, shm_file.shm_fd_, 0);
  SYS_FAIL_LOG_RETURN(MAP_FAILED == physical_ptr_, "mmap");
  size_t space = physical_size_;
  ptr_ = std::align(alignment, size, physical_ptr_, space);
  SYS_FAIL_LOG_RETURN(nullptr == ptr_, "std::align");
  if (!runtime_->shm_map(path, PROT_READ | PROT_WRITE, session_id).ok()) {
    VLOG(1) << "neuron-rtd shm_map failed";
    return;
  }
  VLOG(1) << "allocated shared memory buffer " << path;
  path_ = path;
}

SharedMemoryBuffer::~SharedMemoryBuffer() {
  VLOG(1) << "entering destructor of SharedMemoryBuffer " << path_;
  if (!path_.empty()) {
    TF_LOG_IF_ERROR(runtime_->shm_unmap(path_, PROT_READ | PROT_WRITE));
  }
  if (MAP_FAILED != physical_ptr_) {
    SYS_FAIL_LOG(munmap(physical_ptr_, physical_size_) < 0, "munmap");
  }
}

std::string SharedMemoryBuffer::debug_string() {
  return errors::Internal("SharedMemoryBuffer(ptr=", ptr_, ", size=", size_,
                          ", "
                          "physical_ptr=",
                          physical_ptr_,
                          ", "
                          "physical_size=",
                          physical_size_, ")")
      .error_message();
}

SharedMemoryAllocator::SharedMemoryAllocator()
    : single_allocation_warning_count_(0) {}

Status SharedMemoryAllocator::initialize(const uint64_t session_id,
                                         const std::string& nrtd_address) {
  std::string nrt_shm_map = env_get("NEURON_RTD_SHM_MAP", "");
  if ("no" != nrt_shm_map) {
    session_id_ = session_id;
    runtime_ = std::make_shared<RuntimeGRPC>();
    TF_RETURN_IF_ERROR(runtime_->initialize(nrtd_address));
    size_t id = buffer_vec_.size();
    SharedMemoryPtr shm_ptr = std::make_shared<SharedMemoryBuffer>(
        id, session_id_, /*alignment=*/1, /*size=*/16, runtime_);
    is_valid_ = shm_ptr->is_valid();
    if (!is_valid_) {
      LOG(INFO) << "The current Neuron runtime configuration does not support "
                   "shared memory data transfer. Please refer to "
                   "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/"
                   "neuron-guide/neuron-runtime/"
                   "nrt-theory-of-operation.html#shared-memory-for-inference-"
                   "ifmaps-and-ofmaps "
                   "if you encounter performance problem caused by high CPU "
                   "usage on inf1 instances.";
    }
  }
  return Status::OK();
}

SharedMemoryPtr SharedMemoryAllocator::allocate_shm(const size_t alignment,
                                                    const size_t size) {
  tensorflow::mutex_lock lock(mutex_);
  if (TF_PREDICT_FALSE(!is_valid_)) {
    LOG(ERROR) << "SharedMemoryAllocator is invalid";
  }
  if (size_to_free_buffer_id_.count(size) &&
      size_to_free_buffer_id_[size].size()) {
    // get one from the free buffer set
    std::unordered_set<size_t>* free_buffer_id_set =
        &size_to_free_buffer_id_[size];
    for (size_t free_buffer_id : *free_buffer_id_set) {
      SharedMemoryPtr shm_ptr = buffer_vec_[free_buffer_id];
      if (TF_PREDICT_TRUE(shm_ptr->is_valid())) {
        free_buffer_id_set->erase(free_buffer_id);
        VLOG(1) << "reusing already allocated shm buffer "
                << shm_ptr->debug_string();
        return shm_ptr;
      }
    }
    LOG(ERROR) << "all cached buffers are invalid; returning an invalid buffer";
    auto iter = free_buffer_id_set->begin();
    size_t free_buffer_id = *iter;
    free_buffer_id_set->erase(iter);
    SharedMemoryPtr shm_ptr = buffer_vec_[free_buffer_id];
    VLOG(1) << "reusing already allocated shm buffer "
            << shm_ptr->debug_string();
    return shm_ptr;
  }
  VLOG(1) << "allocating a new shm buffer";
  size_t id = buffer_vec_.size();
  SharedMemoryPtr shm_ptr = std::make_shared<SharedMemoryBuffer>(
      id, session_id_, alignment, size, runtime_);
  buffer_vec_.push_back(shm_ptr);
  ptr_to_id_[shm_ptr->get_ptr()] = id;
  if (TF_PREDICT_FALSE(!shm_ptr->is_valid())) {
    LOG(ERROR) << "allocate_shm failed; " << shm_ptr->debug_string()
               << " will not be available in Neuron runtime";
    return shm_ptr;
  }
  VLOG(1) << "successfully allocated shm buffer " << shm_ptr->debug_string();
  return shm_ptr;
}

void SharedMemoryAllocator::free_shm_unsafe(SharedMemoryPtr shm) {
  if (TF_PREDICT_FALSE(!shm->is_valid())) {
    LOG(ERROR) << "freeing invalid shm buffer " << shm->debug_string();
  }
  VLOG(1) << "freeing shm buf " << shm->get_path();
  size_t size = shm->get_size();
  if (!size_to_free_buffer_id_.count(size)) {
    size_to_free_buffer_id_.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(size),
                                    std::forward_as_tuple());
  }
  size_to_free_buffer_id_[size].insert(shm->get_id());
}

// Individual allocations large than this amount will trigger a warning.
static const double kLargeAllocationWarningThreshold = 0.1;
static const int kMaxSingleAllocationWarnings = 5;

// Cache first invocation to port::AvailableRam, as it can be expensive.
static int64_t LargeAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kLargeAllocationWarningThreshold);
  return value;
}

void* SharedMemoryAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  if ((int64)num_bytes > LargeAllocationWarningBytes() &&
      single_allocation_warning_count_ < kMaxSingleAllocationWarnings) {
    ++single_allocation_warning_count_;
    LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                 << 100 * kLargeAllocationWarningThreshold
                 << "% of system memory.";
  }
  VLOG(1) << "allocating alignment " << alignment << ", num_bytes " << num_bytes
          << " from SharedMemoryAllocator::AllocateRaw";
  return allocate_shm(alignment, num_bytes)->get_ptr();
}

void SharedMemoryAllocator::DeallocateRaw(void* ptr) {
  tensorflow::mutex_lock lock(mutex_);
  if (TF_PREDICT_FALSE(!ptr_to_id_.count(ptr))) {
    LOG(ERROR) << "freeing non-shared-memory pointer " << ptr;
    return;
  }
  size_t id = ptr_to_id_[ptr];
  SharedMemoryPtr shm = buffer_vec_[id];
  free_shm_unsafe(shm);
}

size_t SharedMemoryAllocator::AllocatedSizeSlow(const void* ptr) const {
  if (TF_PREDICT_FALSE(!ptr_to_id_.count(ptr))) {
    LOG(ERROR) << "cannot determine size of non-shared-memory pointer " << ptr;
    return 0;
  }
  size_t id = ptr_to_id_.at(ptr);
  SharedMemoryPtr shm = buffer_vec_[id];
  return shm->get_size();
}

bool SharedMemoryAllocator::is_shm_tensor(const Tensor& tensor) {
  return DataTypeCanUseMemcpy(tensor.dtype()) &&
         ptr_to_id_.count(tensor.tensor_data().data());
}

SharedMemoryPtr SharedMemoryAllocator::get_shm_ptr(const Tensor& tensor) {
  const void* ptr = tensor.tensor_data().data();
  tensorflow::mutex_lock lock(mutex_);
  if (TF_PREDICT_FALSE(!ptr_to_id_.count(ptr))) {
    LOG(ERROR) << "cannot find shm_ptr from non-shared-memory pointer " << ptr;
    return nullptr;
  }
  size_t id = ptr_to_id_.at(ptr);
  return buffer_vec_[id];
}

}  // namespace neuron
}  // namespace tensorflow

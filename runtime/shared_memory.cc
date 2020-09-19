/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/neuron/runtime/macros.h"
#include "tensorflow/neuron/runtime/shared_memory.h"


namespace tensorflow {
namespace neuron {


class ShmFile {
public:
    ShmFile(const std::string &name) {
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
    ShmFile(const ShmFile &shm_file) = delete;
    ShmFile &operator=(const ShmFile &shm_file) = delete;
    ShmFile(ShmFile &&) = delete;
    ShmFile &operator=(ShmFile &&) = delete;
    int shm_fd_ = -1;
private:
    int shm_open_fd_ = -1;
    std::string name_;
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


SharedMemoryBuffer::SharedMemoryBuffer(const size_t id, const size_t size, const uint64_t session_id,
                                       std::shared_ptr<RuntimeGRPC> runtime) : id_(id) {
    VLOG(1) << "entering SharedMemoryBuffer constructor";
    if (nullptr == runtime) {
        LOG(ERROR) << "runtime is not initialized";
        return;
    }
    runtime_ = runtime;
    std::string path = gen_shm_path();
    if (path.empty()) {
        LOG(ERROR) << "cannot generate unique file name for shared memory";
        return;
    }
    ShmFile shm_file(path);
    SYS_FAIL_LOG_RETURN(shm_file.shm_fd_ < 0, "shm_open");
    SYS_FAIL_LOG_RETURN(::ftruncate(shm_file.shm_fd_, size) < 0, "ftruncate");
    ptr_ = static_cast<char*>(::mmap(0, size, PROT_WRITE, MAP_SHARED, shm_file.shm_fd_, 0));
    size_ = size;
    SYS_FAIL_LOG_RETURN(nullptr == ptr_, "mmap");
    if (!runtime_->shm_map(path, PROT_READ | PROT_WRITE, session_id).ok()) {
        VLOG(1) << "neuron-rtd shm_map failed";
        unsupported_by_runtime_ = true;
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
    if (nullptr != ptr_) {
        SYS_FAIL_LOG(munmap(ptr_, size_) < 0, "munmap");
    }
}

SharedMemoryBuffer::SharedMemoryBuffer(const SharedMemoryBuffer &buffer) : id_(buffer.id_) {
    runtime_ = buffer.runtime_;
    ptr_ = buffer.ptr_;
    size_ = buffer.size_;
    path_ = buffer.path_;
}

SharedMemoryBuffer::SharedMemoryBuffer(SharedMemoryBuffer &&buffer) : id_(buffer.id_) {
    runtime_ = buffer.runtime_;
    ptr_ = buffer.ptr_;
    size_ = buffer.size_;
    path_ = buffer.path_;
    buffer.runtime_ = nullptr;
    buffer.ptr_ = nullptr;
    buffer.size_ = 0;
    buffer.path_.clear();
}


SharedMemoryBufferManager::SharedMemoryBufferManager(const uint64_t session_id,
                                                     const std::string &nrtd_address)
                                                     : session_id_(session_id) {
    runtime_ = std::make_shared<RuntimeGRPC>();
    TF_LOG_RETURN_IF_ERROR(runtime_->initialize(nrtd_address));
    is_valid_ = true;
}


SharedMemoryPtr SharedMemoryBufferManager::allocate_shm(const size_t size) {
    if (!is_valid_) {
        VLOG(1) << "SharedMemoryBufferManager is invalid";
        return nullptr;
    }
    tensorflow::mutex_lock lock(mutex_);
    if (!is_valid_) {
        // check once more to kick out threads that have already accessed is_valid_ before the lock
        return nullptr;
    }
    if (size_to_free_buffer_id_.count(size) && size_to_free_buffer_id_[size].size()) {
        // get one from the free buffer set
        VLOG(1) << "getting an already allocated shm buffer";
        std::unordered_set<size_t> *free_buffer_id_set = &size_to_free_buffer_id_[size];
        auto iter = free_buffer_id_set->begin();
        size_t free_buffer_id = *iter;
        free_buffer_id_set->erase(iter);
        return buffer_vec_[free_buffer_id];
    }
    VLOG(1) << "allocating a new shm buffer";
    size_t id = buffer_vec_.size();
    buffer_vec_.push_back(std::make_shared<SharedMemoryBuffer>(id, size, session_id_, runtime_));
    if (!buffer_vec_.back()->is_valid()) {
        if (buffer_vec_.back()->unsupported_by_runtime()) {
            LOG(INFO) << "The current Neuron runtime configuration does not support "
                         "shared memory data transfer. Please refer to "
                         "https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-runtime/nrt-theory-of-operation.md#shared-memory-for-inference-ifmaps-and-ofmaps "
                         "if you encounter performance problem caused by high CPU usage on inf1 instances.";
            is_valid_ = false;
        }
        buffer_vec_.pop_back();
        VLOG(1) << "SharedMemoryBufferManager created an invalid buffer";
        return nullptr;
    }
    return buffer_vec_.back();
}

void SharedMemoryBufferManager::free_shm(SharedMemoryPtr shm) {
    tensorflow::mutex_lock lock(mutex_);
    if (!shm->is_valid()) {
        LOG(ERROR) << "SharedMemoryBufferManager cannot free an invalid shared memory buffer";
        return;
    }
    VLOG(1) << "freeing shm buf " << *shm->get_path();
    size_t size = shm->get_size();
    if (!size_to_free_buffer_id_.count(size)) {
        size_to_free_buffer_id_.emplace(
            std::piecewise_construct, std::forward_as_tuple(size), std::forward_as_tuple());
    }
    size_to_free_buffer_id_[size].insert(shm->get_id());
}

void SharedMemoryBufferManager::clear() {
    tensorflow::mutex_lock lock(mutex_);
    size_to_free_buffer_id_.clear();
    buffer_vec_.clear();
    is_valid_ = false;
}


}  // namespace neuron
}  // namespace tensorflow

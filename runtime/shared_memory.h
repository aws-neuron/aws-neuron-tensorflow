/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_RUNTIME_SHARED_MEMORY_H_
#define TENSORFLOW_NEURON_RUNTIME_SHARED_MEMORY_H_

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/neuron/runtime/runtime_grpc.h"

namespace tensorflow {
namespace neuron {

class SharedMemoryBuffer {
public:
    SharedMemoryBuffer(const size_t id, const size_t size, const uint64_t session_id,
                       std::shared_ptr<RuntimeGRPC> runtime);
    ~SharedMemoryBuffer();
    SharedMemoryBuffer(const SharedMemoryBuffer &);
    SharedMemoryBuffer &operator=(const SharedMemoryBuffer &) = delete;
    SharedMemoryBuffer(SharedMemoryBuffer &&);
    SharedMemoryBuffer &operator=(SharedMemoryBuffer &&) = delete;
    // path_ is assigned when rtd shm_map has returned success
    bool is_valid() { return !path_.empty(); }
    bool unsupported_by_runtime() { return unsupported_by_runtime_; }
    size_t get_id() { return id_; }
    char *get_ptr() { return ptr_; }
    size_t get_size() { return size_; }
    std::string *get_path() { return &path_; }
private:
    const size_t id_;
    std::shared_ptr<RuntimeGRPC> runtime_ = nullptr;
    char *ptr_ = nullptr;
    size_t size_ = 0;
    bool unsupported_by_runtime_ = false;
    std::string path_ = "";
};

typedef std::shared_ptr<SharedMemoryBuffer> SharedMemoryPtr;

class SharedMemoryBufferManager {
public:
    SharedMemoryBufferManager(const uint64_t session_id, const std::string &nrtd_address);
    bool is_valid() { return is_valid_; }
    SharedMemoryPtr allocate_shm(const size_t size);
    void free_shm(SharedMemoryPtr shm);
    void clear();
private:
    tensorflow::mutex mutex_;
    uint64_t session_id_ = RuntimeSession::INVALID_ID;
    std::shared_ptr<RuntimeGRPC> runtime_ = nullptr;
    bool is_valid_ = false;
    std::vector<SharedMemoryPtr> buffer_vec_;
    std::unordered_map<size_t, std::unordered_set<size_t> > size_to_free_buffer_id_;
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_SHARED_MEMORY_H_

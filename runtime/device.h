/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_RUNTIME_DEVICE_H_
#define TENSORFLOW_NEURON_RUNTIME_DEVICE_H_

#include <queue>
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/neuron/runtime/semaphore.h"
#include "tensorflow/neuron/runtime/timestamps.h"
#include "tensorflow/neuron/runtime/profiler.h"
#include "tensorflow/neuron/runtime/tensor_util.h"
#include "tensorflow/neuron/runtime/shared_memory.h"
#include "tensorflow/neuron/runtime/runtime_grpc.h"


namespace tensorflow {
namespace neuron {


typedef std::queue<xla::Semaphore::ScopedReservation> SemResQueue;


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


class NeuronDevice {
public:
    Status initialize(const std::string &nrtd_address,
                      const int num_cores_req, const int num_dup, const uint64_t session_id);
    Status load(uint32_t *nn_id, const StringPiece &executable,
                const uint32_t timeout, const uint32_t ninfer, const bool profile_enabled);
    Status setup_infer_post(RuntimeIO *runtime_io, int64_t post_tag);
    Status post_infer_post(RuntimeIO *runtime_io);
    Status wait_infer_post(RuntimeIO *runtime_io);
    Status setup_infer(RuntimeIO *runtime_io, int64_t post_tag);
    Status post_infer(RuntimeIO *runtime_io);
    Status wait_infer(RuntimeIO *runtime_io);
    Status infer(RuntimeIO *runtime_io, Timestamps *timestamps,
                 ProfilerInterface *profile, const uint32_t nn_id);
    Status infer_post(RuntimeIO *runtime_io, SemResQueue *sem_res_queue,
                      xla::Semaphore *infer_sem, Timestamps *timestamps,
                      const uint32_t nn_id);
    Status infer_wait(RuntimeIO *runtime_io, Timestamps *timestamps);
    void unload(const uint32_t nn_id);
    void acquire_mutex(std::queue<tensorflow::mutex_lock> *mutex_lock_queue);
    Status infer_post_unsafe(RuntimeIO *runtime_io, Timestamps *timestamps,
                             const uint32_t nn_id);
    void clear(bool from_global_state=false);
    size_t num_executable() { return nn_id_to_all_nn_ids_.size(); };
    uint32_t num_cores() { return num_cores_; };
    Status start_model_unsafe(const uint32_t nn_id);
    Status start_ping(const uint32_t nn_id);
    size_t semaphore_factor() { return vec_eg_id_.size(); }
    std::shared_ptr<SharedMemoryBufferManager> shm_buf_mgr_ = nullptr;
private:
    bool is_busy();
    bool running(uint32_t nn_id);
    void set_running(uint32_t nn_id);
    uint32_t nn_get_current_running();
    Status get_active(uint32_t *active_nn_id, const uint32_t nn_id);
    tensorflow::mutex mutex_eg_;
    bool closed_ = false;
    RuntimeGRPC runtime_;
    uint64_t session_id_ = RuntimeSession::INVALID_ID;
    std::vector<uint32_t> vec_eg_id_;
    uint32_t running_nn_id_;
    uint32_t num_cores_ = 0;
    static const size_t EXEC_MAX_CHUNK_SIZE = 1024 * 1024;  // some reasonable number of bytes
    std::string nrtd_address_ = "";
    std::unordered_map<uint32_t, std::vector<uint32_t> > nn_id_to_all_nn_ids_;
    std::unordered_map<uint32_t, size_t> nn_id_to_active_idx_;
};


class NeuronDeviceManager {
public:
    NeuronDeviceManager() {};
    ~NeuronDeviceManager();
    NeuronDeviceManager(const NeuronDeviceManager &) = delete;
    NeuronDeviceManager &operator=(const NeuronDeviceManager &) = delete;
    NeuronDeviceManager(NeuronDeviceManager &&) = delete;
    NeuronDeviceManager &operator=(NeuronDeviceManager &&) = delete;
    Status apply_for_device(
        NeuronDevice **device, const int64_t opt_device_size, const int64_t max_num_duplicates,
        const int64_t device_index=-1);
    void clear_if_empty();
    void clear();
    void clear_from_global_state();
    std::string nrtd_address_;
    static const int64 MAX_NUM_CORES = 64;
    static const int64 MIN_NUM_CORES = 0;
private:
    Status init_default_device(const int64_t opt_device_size, const int64_t max_num_duplicates);
    Status init_devices(const std::vector<int> &num_cores_req_vector,
                        const std::vector<int> &num_dup_vector);
    Status initialize(const int64_t opt_device_size, const int64_t max_num_duplicates);
    tensorflow::mutex global_mutex_;
    static const int DEFAULT_NUM_CORES = -65536;  // any negative number < -MAX_NUM_CORES
    RuntimeSession session_;
    std::array<NeuronDevice, MAX_NUM_CORES> device_array_;
    bool path_set_ = false;
    size_t device_index_ = 0;
    size_t num_devices_ = 0;
    bool ready_ = false;
};


std::string env_get(const char *env_var, const char *default_env_var="");
#define STOI_INVALID_RESULT -65536
int stoi_no_throw(const std::string &str);


}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DEVICE_H_

/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_NEURON_CLIB_DEVICE_H_
#define TENSORFLOW_NEURON_NEURON_CLIB_DEVICE_H_

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

#define NRT_INVALID_NN_ID 0
#define NRT_INVALID_EG_ID 0

#define SYS_FAIL_RETURN(failure_expr, fn_name) {                            \
    if (failure_expr) {                                                     \
        return errors::Internal((fn_name), " failed with errno ", errno);   \
    }                                                                       \
}

#define SYS_FAIL_LOG(failure_expr, fn_name) {                       \
    if (failure_expr) {                                             \
        LOG(ERROR) << (fn_name) << " failed with errno " << errno;  \
    }                                                               \
}


class SharedMemoryManager {
public:
    Status initialize(const std::string &nrtd_address, const uint32_t nn_id,
                      const uint32_t max_num_infers,
                      const std::vector<size_t> &input_tensor_sizes,
                      const std::vector<size_t> &output_tensor_sizes);
    void clear();
    SharedMemory *apply_for_shm();
    void free_shm(SharedMemory *shm);
    std::vector<SharedMemory> shm_vec_;
    bool enabled_ = false;
private:
    Status init_vectors(std::vector<std::string> *names,
                        std::vector<char*> *ptrs,
                        std::vector<size_t> *sizes,
                        std::vector<std::string> *nrt_paths,
                        const std::vector<size_t> &tensor_sizes,
                        const uint32_t nn_id);
    tensorflow::mutex mutex_;
    std::vector<int> shm_busy_vec_;
    size_t num_shms_ = 0;
    RuntimeGRPC runtime_;
};


class NeuronDevice {
public:
    Status initialize(const std::string &nrtd_address,
                      const int num_cores_req, const int num_dup=1);
    Status load(uint32_t *nn_id, const StringPiece &executable,
                const uint32_t timeout, const uint32_t ninfer);
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
    Status init_shm_mgr(SharedMemoryManager **shm_mgr,
                        const uint32_t nn_id, const uint32_t max_num_infers,
                        const std::vector<size_t> input_tensor_sizes,
                        const std::vector<size_t> output_tensor_sizes);
    void clear(bool from_global_state=false);
    size_t num_executable() { return nn_id_to_all_nn_ids_.size(); };
    uint32_t num_cores() { return num_cores_; };
    Status start_model_unsafe(const uint32_t nn_id);
    Status start_ping(const uint32_t nn_id);
    size_t semaphore_factor() { return vec_eg_id_.size(); }
private:
    bool is_busy();
    bool running(uint32_t nn_id);
    void set_running(uint32_t nn_id);
    uint32_t nn_get_current_running();
    Status get_active(uint32_t *active_nn_id, const uint32_t nn_id);
    tensorflow::mutex mutex_eg_;
    bool closed_ = false;
    RuntimeGRPC runtime_;
    std::vector<uint32_t> vec_eg_id_;
    uint32_t running_nn_id_;
    uint32_t num_cores_ = 0;
    static const size_t EXEC_MAX_CHUNK_SIZE = 1024 * 1024;  // some reasonable number of bytes
    std::string nrtd_address_ = "";
    std::unordered_map<uint32_t, SharedMemoryManager> nn_id_to_shm_mgr_;
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
        NeuronDevice **device, int64_t opt_device_size, int64_t device_index=-1);
    void clear_if_empty();
    void clear();
    void clear_from_global_state();
    std::string nrtd_address_;
    static const int64 MAX_NUM_CORES = 64;
    static const int64 MIN_NUM_CORES = 0;
private:
    Status init_default_device(int64_t opt_device_size);
    Status init_devices(const std::vector<int> &num_cores_req_vector,
                        const std::vector<int> &num_dup_vector=std::vector<int>());
    Status initialize(int64_t opt_device_size);
    tensorflow::mutex global_mutex_;
    static const int DEFAULT_NUM_CORES = -65536;  // any negative number < -MAX_NUM_CORES
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

#endif  // TENSORFLOW_NEURON_NEURON_CLIB_DEVICE_H_

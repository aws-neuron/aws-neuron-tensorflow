/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_NEURON_CLIB_DEVICE_H_
#define TENSORFLOW_NEURON_NEURON_CLIB_DEVICE_H_

#include <queue>
#include "tensorflow/core/platform/mutex.h"
#include "semaphore.h"
#include "timestamps.h"
#include "profiler.h"
#include "tensor_util.h"
#include "shared_memory.h"
#include "runtime_grpc.h"


namespace tensorflow {
namespace neuron {


typedef std::queue<xla::Semaphore::ScopedReservation> SemResQueue;

#define NRT_INVALID_NN_ID 0

#define SYS_FAIL_RETURN(failure_expr, fn_name) {                            \
    if (failure_expr) {                                                     \
        return errors::Unknown((fn_name), " failed with errno ", errno);    \
    }                                                                       \
}

#define SYS_FAIL_LOG(failure_expr, fn_name) {                       \
    if (failure_expr) {                                             \
        LOG(ERROR) << (fn_name) << " failed with errno " << errno;  \
    }                                                               \
}


class SharedMemoryManager {
public:
    SharedMemoryManager() {}
    ~SharedMemoryManager();
    Status initialize(const std::string &nrtd_address, const uint32_t nn_id,
                      const std::vector<size_t> &input_tensor_sizes,
                      const std::vector<size_t> &output_tensor_sizes);
    SharedMemory shm_;
private:
    Status init_vectors(std::vector<std::string> *names,
                        std::vector<void*> *ptrs,
                        std::vector<size_t> *sizes,
                        std::vector<std::string> *nrt_paths,
                        const std::vector<size_t> &tensor_sizes,
                        const uint32_t nn_id);
    RuntimeGRPC runtime_;
    std::vector<std::string> nrt_input_paths_;
    std::vector<std::string> nrt_output_paths_;
};


class NeuronDevice {
public:
    NeuronDevice() {};
    Status initialize(const std::string &nrtd_address, const int num_cores_req);
    Status load(uint32_t *nn_id, const StringPiece &executable,
                const uint32_t timeout, const uint32_t ninfer);
    Status infer(std::vector<Tensor*> *output_tensors, Timestamps *timestamps,
                 ProfilerInterface *profile, const uint32_t nn_id,
                 AttrList &input_names, AttrList &output_names,
                 const std::vector<const Tensor*> &input_tensors,
                 const SharedMemory &shm);
    Status infer_post(NMGROutputs *nmgr_outputs, SemResQueue *sem_res_queue,
                      xla::Semaphore *infer_sem, Timestamps *timestamps,
                      const uint32_t nn_id, AttrList &input_names,
                      const std::vector<const Tensor*> &input_tensors);
    Status infer_wait(NMGROutputs *nmgr_outputs, Timestamps *timestamps,
                      AttrList &output_names);
    void unload(const uint32_t nn_id);
    void acquire_mutex(std::queue<tensorflow::mutex_lock> *mutex_lock_queue);
    Status infer_post_unsafe(NMGROutputs *nmgr_outputs, Timestamps *timestamps,
                             const uint32_t nn_id, AttrList &input_names,
                             const std::vector<const Tensor*> &input_tensors);
    void clear();
    size_t num_executable() { return nn_id_set_.size(); };
    uint32_t num_cores() { return num_cores_; };
private:
    Status start_model(const uint32_t nn_id);
    bool is_busy();
    bool running(uint32_t nn_id);
    void set_running(uint32_t nn_id);
    uint32_t nn_get_current_running();
    tensorflow::mutex mutex_eg_;
    RuntimeGRPC runtime_;
    bool create_eg_done_ = false;
    uint32_t eg_id_;
    uint32_t running_nn_id_;
    std::set<uint32_t> nn_id_set_;
    uint32_t num_cores_ = 0;
    static const size_t EXEC_MAX_CHUNK_SIZE = 1024 * 1024;  // some reasonable number of bytes
    std::string nrtd_address_ = "";
};


class NeuronDeviceManager {
public:
    NeuronDeviceManager() {};
    Status apply_for_device(NeuronDevice **device, int64_t opt_device_size);
    void clear_if_empty();
    void clear();
    ~NeuronDeviceManager();
    std::string nrtd_address_;
    static const int64 MAX_NUM_CORES = 64;
    static const int64 MIN_NUM_CORES = 0;
private:
    Status init_default_device(int64_t opt_device_size);
    Status init_devices(const std::vector<int> &num_cores_req_vector);
    Status initialize(int64_t opt_device_size);
    tensorflow::mutex global_mutex_;
    static const int DEFAULT_NUM_CORES = -1;  // any negative number
    std::array<NeuronDevice, MAX_NUM_CORES> device_array_;
    bool path_set_ = false;
    size_t device_index_ = 0;
    size_t num_devices_ = 0;
    bool ready_ = false;
};


std::string env_get(const char *env_var, const char *default_env_var="");
int stoi_no_throw(const std::string &str);


}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_NEURON_CLIB_DEVICE_H_

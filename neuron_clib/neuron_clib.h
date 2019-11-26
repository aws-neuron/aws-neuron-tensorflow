/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_CONTRIB_KAENA_KAENA_CLIB_H_
#define TENSORFLOW_CONTRIB_KAENA_KAENA_CLIB_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "nerr.pb.h"
#include "nmgr_service.pb.h"
#include "nmgr_service.grpc.pb.h"


namespace tensorflow {
namespace kaena {


#define NRT_INVALID_NN_ID 0

#define NRT_CHECK_RETURN(fn_name, status, response) {                       \
    if (!((status).ok() && 0 == (response).status().code())) {              \
        return nrt_error_status((fn_name), (status), (response).status());  \
    }                                                                       \
}

#define NRT_CHECK_LOG(fn_name, status, response) {                      \
    if (!((status).ok() && 0 == (response).status().code())) {          \
        LOG(ERROR) << nrt_error_status(                                 \
            (fn_name), (status), (response).status()).error_message();  \
    }                                                                   \
}

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

inline Status nrt_error_status(const std::string &fn_name,
                               const grpc::Status &status,
                               const nrt::status &nrt_status) {
    return errors::Internal(
        "nrt::", fn_name, " failed with grpc status code ", status.error_code(),
        ", error message \"", status.error_message(), "\"; nrt status code ",
        nrt_status.code(), ", details \"", nrt_status.details(), "\""
    );
}


class SharedMemory {
public:
    SharedMemory(size_t size) : size_(size) {}
    Status initialize(const std::unique_ptr<nrt::nmgr_v1::Stub> &stub, uint32_t nn_id);
    const std::string name() { return name_; }
    const size_t size() { return size_; }
    void *ptr() { return ptr_; }
    void clear(const std::unique_ptr<nrt::nmgr_v1::Stub> &stub);
private:
    bool shm_open_done_ = false;
    bool shm_map_done_ = false;
    void *ptr_ = nullptr;
    std::string name_ = "";
    const size_t size_ = 0;
};


class SharedMemoryAllocator : public Allocator {
public:
    SharedMemoryAllocator(SharedMemory *shared_memory);
    ~SharedMemoryAllocator() override;
    std::string Name() override;
    void *AllocateRaw(size_t alignment, size_t num_bytes) override;
    void DeallocateRaw(void *ptr) override;
private:
    SharedMemory *shared_memory_;
};


class NeuronDevice {
public:
    NeuronDevice() {};
    Status initialize(std::unique_ptr<nrt::nmgr_v1::Stub> &stub,
                      const std::string &nrtd_address, int num_cores_req);
    void clear(std::unique_ptr<nrt::nmgr_v1::Stub> &stub);
    uint32_t eg_id() { return eg_id_; };
    size_t num_executable() { return nn_id_set_.size(); };
    uint32_t num_cores() { return num_cores_; };
    void register_executable(uint32_t nn_id) { nn_id_set_.insert(nn_id); };
    void deregister_executable(uint32_t nn_id) { nn_id_set_.erase(nn_id); };
    std::unordered_map<void*, SharedMemory*> *get_ptr2shm() { return &ptr2shm_; };
    Status is_valid();
    bool is_busy();
    bool running(uint32_t nn_id);
    void set_running(uint32_t nn_id);
    uint32_t nn_get_current_running();
    tensorflow::mutex mutex_eg_;
private:
    bool create_eg_done_ = false;
    uint32_t eg_id_;
    uint32_t running_nn_id_;
    std::unordered_map<void*, SharedMemory*> ptr2shm_;
    std::set<uint32_t> nn_id_set_;
    uint32_t num_cores_ = 0;
};


class NeuronDeviceManager {
public:
    NeuronDeviceManager() {};
    Status initialize(int64_t opt_device_size);
    bool ready() { return ready_; };
    NeuronDevice *get_device();
    bool is_empty();
    void clear();
    ~NeuronDeviceManager();
    tensorflow::mutex global_mutex_;
    static const int64 MAX_NUM_CORES = 64;
    static const int64 MIN_NUM_CORES = 0;
private:
    Status init_default_device(const std::string &nrtd_address, int64_t opt_device_size);
    Status init_devices(const std::vector<int> &num_cores_req_vector,
                        const std::string &nrtd_address);
    std::unique_ptr<nrt::nmgr_v1::Stub> stub_;
    static const int DEFAULT_NUM_CORES = -1;  // any negative number
    std::array<NeuronDevice, MAX_NUM_CORES> device_array_;
    size_t device_index_ = 0;
    size_t num_devices_ = 0;
    bool ready_ = false;
};


class FALTimestamps {
public:
    void mark_enter() { enter_ = now(); };
    void mark_above_nrtd_infer() { above_nrtd_infer_ = now(); };
    void mark_below_nrtd_infer() { below_nrtd_infer_ = now(); };
    void mark_exit() { exit_ = now(); };
    std::string timing_string();
private:
    uint64 enter_ = 0;
    uint64 above_nrtd_infer_ = 0;
    uint64 below_nrtd_infer_ = 0;
    uint64 exit_ = 0;

    std::string time_unit_ = " us";
    uint64 now() { return Env::Default()->NowMicros(); };
};


std::string env_get(const char *env_var, const char *default_env_var="");
int stoi_no_throw(const std::string &str);


}  // namespace kaena
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_KAENA_KAENA_CLIB_H_

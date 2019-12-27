/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_NEURON_CLIB_NEURON_CLIB_H_
#define TENSORFLOW_NEURON_NEURON_CLIB_NEURON_CLIB_H_

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "timestamps.h"
#include "nmgr_service.grpc.pb.h"


namespace tensorflow {
namespace neuron {


#define NRT_INVALID_NN_ID 0

#define NRT_GRPC(func, request, response) ({                    \
    grpc::Status status;                                        \
    grpc::ClientContext context;                                \
    status = (func)(&context, (request), (response));           \
    if (grpc::StatusCode::UNAVAILABLE == status.error_code()) { \
        grpc::ClientContext context;                            \
        status = (func)(&context, (request), (response));       \
    }                                                           \
    status;                                                     \
})

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


class SharedMemoryManager {
public:
    SharedMemoryManager() {}
    ~SharedMemoryManager();
    Status initialize(const std::string &nrtd_address, const uint32_t nn_id,
                      const std::vector<size_t> &input_tensor_sizes,
                      const std::vector<size_t> &output_tensor_sizes);
    bool enabled_ = false;
    std::vector<std::string> input_names_;
    std::vector<void*> input_ptrs_;
    std::vector<size_t> input_sizes_;
    std::vector<std::string> output_names_;
    std::vector<void*> output_ptrs_;
    std::vector<size_t> output_sizes_;
private:
    Status init_vectors(std::vector<std::string> *names,
                        std::vector<void*> *ptrs,
                        std::vector<size_t> *sizes,
                        std::vector<std::string> *grpc_names,
                        const std::vector<size_t> &tensor_sizes,
                        const uint32_t nn_id);
    void nrt_shm_unmap(const std::string &name);
    std::unique_ptr<nrt::nmgr_v1::Stub> stub_;
    std::vector<std::string> input_grpc_names_;
    std::vector<std::string> output_grpc_names_;
};


class NeuronDevice {
public:
    NeuronDevice() {};
    Status initialize(const std::string &nrtd_address, int num_cores_req);
    Status load(uint32_t *nn_id, const StringPiece &executable,
                const uint32_t timeout, const uint32_t ninfer);
    void unload(const uint32_t nn_id);
    void clear();
    size_t num_executable() { return nn_id_set_.size(); };
    uint32_t num_cores() { return num_cores_; };
    Status is_valid();
    bool is_busy();
    bool running(uint32_t nn_id);
    void set_running(uint32_t nn_id);
    uint32_t nn_get_current_running();
    tensorflow::mutex mutex_eg_;
private:
    std::unique_ptr<nrt::nmgr_v1::Stub> stub_;
    bool create_eg_done_ = false;
    uint32_t eg_id_;
    uint32_t running_nn_id_;
    std::set<uint32_t> nn_id_set_;
    uint32_t num_cores_ = 0;
    static const size_t EXEC_MAX_CHUNK_SIZE = 1024 * 1024;  // some reasonable number of bytes
};


class NeuronDeviceManager {
public:
    NeuronDeviceManager() {};
    Status apply_for_device(NeuronDevice **device, int64_t opt_device_size);
    Status clear_if_empty();
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
    std::unique_ptr<nrt::nmgr_v1::Stub> stub_;
    static const int DEFAULT_NUM_CORES = -1;  // any negative number
    std::array<NeuronDevice, MAX_NUM_CORES> device_array_;
    bool path_set_ = false;
    size_t device_index_ = 0;
    size_t num_devices_ = 0;
    bool ready_ = false;
};


std::string env_get(const char *env_var, const char *default_env_var="");
int stoi_no_throw(const std::string &str);
Status init_stub(std::unique_ptr<nrt::nmgr_v1::Stub> *stub,
                 const std::string &nrtd_address);


}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_NEURON_CLIB_NEURON_CLIB_H_

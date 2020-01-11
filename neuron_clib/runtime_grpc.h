/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_NEURON_CLIB_RUNTIME_GRPC_H_
#define TENSORFLOW_NEURON_NEURON_CLIB_RUNTIME_GRPC_H_

#include "timestamps.h"
#include "tensor_util.h"
#include "shared_memory.h"
#include "nmgr_service.grpc.pb.h"


namespace tensorflow {
namespace neuron {

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

inline Status nrt_error_status(const std::string &fn_name,
                               const grpc::Status &status,
                               const nrt::status &nrt_status) {
    return errors::Internal(
        "nrt::", fn_name, " failed with grpc status code ", status.error_code(),
        ", error message \"", status.error_message(), "\"; nrt status code ",
        nrt_status.code(), ", details \"", nrt_status.details(), "\""
    );
}

class NMGROutputs {
public:
    NMGROutputs() {};
    Status initialize(std::vector<Tensor*> *output_tensors,
                      const uint32_t nn_id, AttrList &output_names) {
        return Status::OK();
    }
    ~NMGROutputs() {};
    NMGROutputs(const NMGROutputs &nmgr_outputs) {
        cookie = nmgr_outputs.cookie;
    }
    NMGROutputs &operator=(const NMGROutputs &nmgr_outputs) {
        cookie = nmgr_outputs.cookie;
        return *this;
    }
    uint64_t cookie = 0;
};

class RuntimeGRPC {
public:
    RuntimeGRPC() {};
    Status initialize(const std::string &nrtd_address);
    Status create_eg(uint32_t *eg_id, uint32_t *num_cores, const int num_cores_req);
    Status load(uint32_t *nn_id, const uint32_t eg_id, const StringPiece &executable,
                const uint32_t timeout, const uint32_t ninfer);
    Status start(const uint32_t nn_id);
    Status infer(std::vector<Tensor*> *output_tensors, Timestamps *timestamps,
                 const uint32_t nn_id,
                 AttrList &input_names, AttrList &output_names,
                 const std::vector<const Tensor*> &input_tensors,
                 const SharedMemory &shm);
    Status infer_post(NMGROutputs *nmgr_outputs, Timestamps *timestamps,
                      const uint32_t nn_id, AttrList &input_names,
                      const std::vector<const Tensor*> &input_tensors);
    Status infer_wait(std::vector<Tensor*> *output_tensors, Timestamps *timestamps,
                      const NMGROutputs &nmgr_outputs, AttrList &output_names);
    Status stop(const uint32_t nn_id);
    Status unload(const uint32_t nn_id);
    Status destroy_eg(const uint32_t eg_id);
    Status shm_map(const std::string &path, const uint32_t mmap_prot);
    Status shm_unmap(const std::string &path, const uint32_t mmap_prot);
private:
    std::unique_ptr<nrt::nmgr_v1::Stub> stub_;
    static const size_t EXEC_MAX_CHUNK_SIZE = 1024 * 1024;  // some reasonable number of bytes
    std::string nrtd_address_ = "";
};

Status copy_output_tensors(std::vector<Tensor*> *output_tensors,
                           const nrt::infer_response &response,
                           AttrList &output_names);

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_NEURON_CLIB_RUNTIME_GRPC_H_

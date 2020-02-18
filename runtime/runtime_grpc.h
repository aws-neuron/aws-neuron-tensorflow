/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#ifndef TENSORFLOW_NEURON_NEURON_CLIB_RUNTIME_GRPC_H_
#define TENSORFLOW_NEURON_NEURON_CLIB_RUNTIME_GRPC_H_

#include "tensorflow/neuron/runtime/timestamps.h"
#include "tensorflow/neuron/runtime/tensor_util.h"
#include "tensorflow/neuron/runtime/shared_memory.h"
#include "tensorflow/neuron/runtime/proto/nmgr_service.grpc.pb.h"


namespace tensorflow {
namespace neuron {

#define NRT_INVALID_COOKIE 0
#define INFER_POST_INVALID_TAG -1

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

#define NRT_CHECK_RETURN(fn_name, grpc_status, response) {                      \
    if (!((grpc_status).ok() && 0 == (response).status().code())) {             \
        return nrt_error_status((fn_name), (grpc_status), (response).status()); \
    }                                                                           \
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

class RuntimeIO {
public:
    Status setup(AttrList &input_names, const std::vector<const Tensor*> &input_tensors,
                 AttrList &output_names, const std::vector<Tensor*> &output_tensors,
                 const uint32_t nn_id, SharedMemory *shm=nullptr);
    Status finish();
    uint64_t cookie = NRT_INVALID_COOKIE;
    SharedMemory *shm_ = nullptr;
    nrt::infer_request request_;
    nrt::infer_post_response post_response_;
    grpc::Status post_status_;
    int64_t post_tag_ = INFER_POST_INVALID_TAG;
    nrt::infer_wait_request wait_request_;
    nrt::infer_response response_;
    AttrList *output_names_;
    std::vector<Tensor*> output_tensors_;
    grpc::ClientContext context_;
    grpc::CompletionQueue cq_;
    std::unique_ptr<grpc::ClientAsyncResponseReader<nrt::infer_post_response> > rpc_infer_post_ = nullptr;
    std::unique_ptr<grpc::ClientAsyncResponseReader<nrt::infer_response> > rpc_infer_ = nullptr;
};

class RuntimeGRPC {
public:
    Status initialize(const std::string &nrtd_address);
    Status create_eg(uint32_t *eg_id, uint32_t *num_cores, const int num_cores_req);
    Status load(uint32_t *nn_id, const uint32_t eg_id, const StringPiece &executable,
                const uint32_t timeout, const uint32_t ninfer);
    Status start(const uint32_t nn_id);
    Status start_ping(const uint32_t nn_id);
    Status setup_infer_post(RuntimeIO *runtime_io, int64_t post_tag);
    Status post_infer_post(RuntimeIO *runtime_io);
    Status wait_infer_post(RuntimeIO *runtime_io);
    Status setup_infer(RuntimeIO *runtime_io, int64_t post_tag);
    Status post_infer(RuntimeIO *runtime_io);
    Status wait_infer(RuntimeIO *runtime_io);
    Status infer_post(RuntimeIO *runtime_io);
    Status infer_wait(RuntimeIO *runtime_io);
    Status stop(const uint32_t nn_id);
    Status unload(const uint32_t nn_id, bool from_global_state=false);
    Status destroy_eg(const uint32_t eg_id, bool from_global_state=false);
    Status shm_map(const std::string &path, const uint32_t mmap_prot);
    Status shm_unmap(const std::string &path, const uint32_t mmap_prot);
private:
    std::unique_ptr<nrt::nmgr_v1::Stub> stub_;
    static const size_t EXEC_MAX_CHUNK_SIZE = 1024 * 1024;  // some reasonable number of bytes
    std::string nrtd_address_ = "";
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_NEURON_CLIB_RUNTIME_GRPC_H_

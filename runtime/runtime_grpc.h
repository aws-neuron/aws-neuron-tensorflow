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

#ifndef TENSORFLOW_NEURON_RUNTIME_RUNTIME_GRPC_H_
#define TENSORFLOW_NEURON_RUNTIME_RUNTIME_GRPC_H_

#include "macros.h"
#include "tensor_util.h"
#include "shared_memory_io.h"
#include "nmgr_service.grpc.pb.h"
#include "nmgr_session_service.grpc.pb.h"
#include "nerr.pb.h"


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
    nrt::status nrtd_status = (response).status();                              \
    if (!((grpc_status).ok() && nrt::nerr::NERR_OK == nrtd_status.code())) {    \
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

template <class T_request, class T_response>
class RuntimeSwitcher {
public:
    RuntimeSwitcher() {}
    T_request request_;
    T_response response_;
    grpc::Status status_;
    grpc::ClientContext context_;
    grpc::CompletionQueue cq_;
    std::unique_ptr<grpc::ClientAsyncResponseReader<T_response> > rpc_ = nullptr;
    int64_t post_tag_ = -1;
private:
    TFN_DISALLOW_COPY_MOVE_ASSIGN(RuntimeSwitcher);
};
typedef RuntimeSwitcher<nrt::start_request, nrt::start_response> RuntimeStarter;
typedef RuntimeSwitcher<nrt::stop_request, nrt::stop_response> RuntimeStopper;

class RuntimeIO {
public:
    RuntimeIO() {}
    Status setup(AttrList &input_names, AttrList &output_names,
                 const std::vector<Tensor*> &output_tensors,
                 const uint32_t nn_id, thread::ThreadPool *thread_pool=nullptr,
                 SharedMemory *shm=nullptr);
    Status copy_input_tensors(const std::vector<const Tensor*> &input_tensors);
    void set_nn_id(const uint32_t nn_id) { request_.mutable_h_nn()->set_id(nn_id); }
    uint32_t get_nn_id() { return request_.mutable_h_nn()->id(); }
    Status finish();
private:
    friend class RuntimeGRPC;
    uint64_t cookie = NRT_INVALID_COOKIE;
    thread::ThreadPool *thread_pool_;
    nrt::infer_request request_;
    nrt::infer_post_response post_response_;
    grpc::Status post_status_;
    int64_t post_tag_ = INFER_POST_INVALID_TAG;
    nrt::infer_wait_request wait_request_;
    nrt::infer_response response_;
    AttrList *output_names_;
    std::vector<Tensor*> output_tensors_;
    bool use_shm_ = false;
    std::vector<void*> input_ptrs_;
    std::vector<void*> output_ptrs_;
    TFN_DISALLOW_COPY_MOVE_ASSIGN(RuntimeIO);
};

class RuntimeGRPC {
public:
    RuntimeGRPC() {}
    Status initialize(const std::string &nrtd_address);
    Status create_eg(uint32_t *eg_id, uint32_t *num_cores, const int num_cores_req,
                     const uint64_t session_id);
    Status load(uint32_t *nn_id, const uint32_t eg_id, const StringPiece &executable,
                const uint32_t timeout, const uint32_t ninfer, const bool profile_enabled,
                const uint64_t session_id);
    Status post_start(RuntimeStarter *starter, const uint32_t nn_id);
    Status wait_start(RuntimeStarter *starter);
    Status infer_post(RuntimeIO *runtime_io);
    Status infer_wait(RuntimeIO *runtime_io);
    Status stop(const uint32_t nn_id);
    Status post_stop(RuntimeStopper *stopper, const uint32_t nn_id);
    Status wait_stop(RuntimeStopper *stopper);
    Status unload(const uint32_t nn_id, bool from_global_state=false);
    Status destroy_eg(const uint32_t eg_id, bool from_global_state=false);
    Status shm_map(const std::string &path, const uint32_t mmap_prot, const uint64_t session_id);
    Status shm_unmap(const std::string &path, const uint32_t mmap_prot);
private:
    std::unique_ptr<nrt::nmgr_v1::Stub> stub_;
    static const size_t EXEC_MAX_CHUNK_SIZE = 1024 * 1024;  // some reasonable number of bytes
    std::string nrtd_address_ = "";
    TFN_DISALLOW_COPY_MOVE_ASSIGN(RuntimeGRPC);
};

class RuntimeSession {
public:
    RuntimeSession();
    Status initialize(const std::string& nrtd_address);
    ~RuntimeSession();
    uint64_t get_id() { return id_; }
    static const uint64_t INVALID_ID = 0;
private:
    typedef grpc::ClientReaderWriter<nrt::session_monitor_request, nrt::session_monitor_response>
        SessionReaderWriter;
    uint64_t id_ = INVALID_ID;
    std::unique_ptr<nrt::nmgr_session_manager::Stub> stub_;
    std::shared_ptr<grpc::ClientContext> context_;
    std::shared_ptr<SessionReaderWriter> stream_;
    TFN_DISALLOW_COPY_MOVE_ASSIGN(RuntimeSession);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_RUNTIME_GRPC_H_

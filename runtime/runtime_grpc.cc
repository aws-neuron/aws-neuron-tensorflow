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

#include <grpcpp/grpcpp.h>
#include "nmgr.pb.h"
#include "runtime_grpc.h"


namespace tensorflow {
namespace neuron {


Status RuntimeIO::setup(
        AttrList &input_names, const std::vector<const Tensor*> &input_tensors,
        AttrList &output_names, const std::vector<Tensor*> &output_tensors,
        const uint32_t nn_id, thread::ThreadPool *thread_pool, SharedMemory *shm) {
    thread_pool_ = thread_pool;
    if (nullptr != shm) {
        if (input_names.s_size() > (int64)shm->input_ptrs_.size()) {
            return errors::Aborted("shared memory is invalid");
        }
        use_shm_ = true;
        output_ptrs_ = shm->output_ptrs_;
    }
    for (auto idx = 0; idx < input_names.s_size(); ++idx) {
        nrt::infer_io *infer_io = request_.add_ifmap();
        infer_io->set_name(input_names.s(idx));
        StringPiece tensor_data(input_tensors[idx]->tensor_data());
        if (use_shm_) {
            infer_io->mutable_buf_shm()->set_path(*shm->input_paths_[idx]);
            fast_memcpy(thread_pool_, shm->input_ptrs_[idx], tensor_data.data(), tensor_data.size());
        } else {
            infer_io->set_buf(tensor_data.data(), tensor_data.size());
        }
    }
    if (use_shm_) {
        for (int idx = 0; idx < output_names.s_size(); ++idx) {
            nrt::infer_io *infer_io = request_.add_shm_ofmap();
            infer_io->set_name(output_names.s(idx));
            infer_io->mutable_buf_shm()->set_path(*shm->output_paths_[idx]);
        }
        for (int idx = 0; idx < output_names.s_size(); ++idx) {
            nrt::infer_io *infer_io = wait_request_.add_shm_ofmap();
            infer_io->set_name(output_names.s(idx));
            infer_io->mutable_buf_shm()->set_path(*shm->output_paths_[idx]);
        }
    }
    request_.mutable_h_nn()->set_id(nn_id);
    output_names_ = &output_names;
    output_tensors_ = output_tensors;
    return Status::OK();
}

Status RuntimeIO::finish() {
    std::vector<StringPiece> raw_output_tensors;
    if (use_shm_) {
        if (output_names_->s_size() > (int64)output_ptrs_.size()) {
            return errors::Aborted("shared memory is invalid");
        }
    }
    if (!use_shm_) {
        std::unordered_map<std::string, StringPiece> map_name_raw;
        for (const auto &infer_io : response_.ofmap()) {
            map_name_raw.emplace(infer_io.name(), infer_io.buf());
        }
        for (auto idx = 0; idx < output_names_->s_size(); ++idx) {
            if (map_name_raw.find(output_names_->s(idx)) == map_name_raw.end()) {
                return errors::NotFound(
                    "tensor name", output_names_->s(idx), " not found in infer_response.ofmap()");
            }
            raw_output_tensors.push_back(map_name_raw[output_names_->s(idx)]);
        }
        for (auto idx = 0; idx < output_names_->s_size(); ++idx) {
            StringPiece out_tensor_raw = raw_output_tensors[idx];
            Tensor *out_tensor = output_tensors_[idx];
            TF_RETURN_WITH_CONTEXT_IF_ERROR(
                tensor_memcpy(thread_pool_, out_tensor, out_tensor_raw),
                "tensor_memcpy failure on tensor name: ", output_names_->s(idx));
        }
        return Status::OK();
    }
    for (auto idx = 0; idx < output_names_->s_size(); ++idx) {
        StringPiece out_tensor_raw;
        if (use_shm_) {
            size_t size = output_tensors_[idx]->tensor_data().size();
            out_tensor_raw = StringPiece(output_ptrs_[idx], size);
        } else {
            out_tensor_raw = raw_output_tensors[idx];
        }
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            tensor_memcpy(thread_pool_, output_tensors_[idx], out_tensor_raw),
            "tensor_memcpy failure on tensor name: ", output_names_->s(idx));
    }
    return Status::OK();
}


Status RuntimeGRPC::initialize(const std::string &nrtd_address) {
    nrtd_address_ = nrtd_address;
    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    ch_args.SetMaxSendMessageSize(-1);
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        nrtd_address, grpc::InsecureChannelCredentials(), ch_args);
    if (!channel) {
        return errors::Unavailable(
            "cannot establish grpc channel to neuron-rtd server");
    }
    stub_ = nrt::nmgr_v1::NewStub(channel);
    if (!stub_) {
        return errors::Unavailable("cannot create stub");
    }
    return Status::OK();
}

Status RuntimeGRPC::create_eg(uint32_t *eg_id, uint32_t *num_cores,
                              const int num_cores_req, const uint64_t session_id) {
    nrt::create_eg_request request;
    if (num_cores_req >= 0) {
        request.set_nc_count((uint32_t)num_cores_req);
    }
    if (RuntimeSession::INVALID_ID != session_id) {
        request.set_session_id(session_id);
    }
    nrt::create_eg_response response;
    grpc::Status status = NRT_GRPC(stub_->create_eg, request, &response);
    if (status.ok() && nrt::nerr::NERR_RESOURCE_NC == response.status().code()) {
        return errors::ResourceExhausted(
            "All machine learning accelerators are currently being consumed. ",
            "Please check if there are other processes running ",
            "on the accelerator. If no other processes are consuming machine ",
            "learning accelerator resource, please manually free up hardware ",
            "resource by `sudo systemctl restart neuron-rtd`. ",
            "If you have package `aws-neuron-tools` installed, you may also "
            "free up resource by `/opt/aws/neuron/bin/neuron-cli reset`. ",
            "IMPORTANT: MANUALLY FREEING UP HARDWARE RESOURCE CAN DESTROY ",
            "YOUR OTHER PROCESSES RUNNING ON MACHINE LEARNING ACCELERATORS!"
        );
    }
    NRT_CHECK_RETURN("create_eg", status, response);
    *eg_id = response.h_eg().id();
    *num_cores = response.nc_count();
    return Status::OK();
}

Status RuntimeGRPC::load(
        uint32_t *nn_id, const uint32_t eg_id, const StringPiece &executable,
        const uint32_t timeout, const uint32_t ninfer, const bool profile_enabled,
        const uint64_t session_id) {
    // load
    grpc::ClientContext context;
    nrt::load_response response;
    std::unique_ptr<grpc::ClientWriter<nrt::load_request> > writer(
        stub_->load(&context, &response));
    nrt::load_request request;

    #define WRITE_LOAD_REQUEST {                                                \
        if (!writer->Write(request)) {                                          \
            return errors::DataLoss("neuron-rtd load failure - broken stream"); \
        }                                                                       \
    }
    // eg_id
    request.mutable_h_eg()->set_id(eg_id);
    WRITE_LOAD_REQUEST;

    // session_id
    if (RuntimeSession::INVALID_ID != session_id) {
        request.set_session_id(session_id);
        WRITE_LOAD_REQUEST;
    }

    // neff_size
    size_t exec_total_size = executable.size();
    request.set_neff_size(exec_total_size);
    WRITE_LOAD_REQUEST;

    // model_params
    nrt::model_params *model_params = request.mutable_model_params();
    model_params->mutable_timeout()->set_data(timeout);
    model_params->mutable_ninfer()->set_data(ninfer);
    if (profile_enabled) {
        model_params->mutable_enable_tracing()->set_data(1);
        model_params->mutable_enable_node_profiling()->set_data(1);
    }
    WRITE_LOAD_REQUEST;

    // neff file content
    for (size_t pos = 0; pos < exec_total_size; pos += EXEC_MAX_CHUNK_SIZE) {
        size_t remaining = exec_total_size - pos;
        size_t chunk_size = std::min(remaining, EXEC_MAX_CHUNK_SIZE);
        StringPiece file_chunk = executable.substr(pos, chunk_size);
        request.mutable_neff_chunk()->set_chunk(file_chunk.data(), chunk_size);
        WRITE_LOAD_REQUEST;
    }
    if (!writer->WritesDone()) {
        return errors::DataLoss("neuron-rtd load failure - broken stream");
    }
    grpc::Status status = writer->Finish();
    if (status.ok() && nrt::nerr::NERR_OK != response.status().code()) {
        bool unsupp(nrt::nerr::NERR_UNSUPPORTED_VERSION == response.status().code());
        bool invalid(nrt::nerr::NERR_INVALID == response.status().code());
        bool unsupp_parsed(
            response.status().details().find("NEFF version mismatch") != std::string::npos);
        if (unsupp || (invalid && (unsupp_parsed || "" == response.status().details()))) {
            return errors::InvalidArgument(
                "Neuron runtime could not load the compiled SavedModel provided because it is "
                "compiled with a neuron-cc that is not compatible with the runtime version used. "
                "Please make sure you upgrade the Neuron runtime version used. "
                "(neuron-rtd error code: ",
                response.status().code(), ", details: ", response.status().details(), ")");
        }
    }
    NRT_CHECK_RETURN("load", status, response);
    *nn_id = response.h_nn().id();
    return Status::OK();
}

Status RuntimeGRPC::start(const uint32_t nn_id) {
    nrt::start_request request;
    request.mutable_h_nn()->set_id(nn_id);
    nrt::start_response response;
    grpc::Status status = NRT_GRPC(stub_->start, request, &response);
    NRT_CHECK_RETURN("start", status, response);
    return Status::OK();
}

static Status wait_grpc_cq(void **got_tag, bool *ok, grpc::CompletionQueue *cq, const int64_t post_tag) {
    if (!cq->Next(got_tag, ok)) {
        return errors::Internal("CompletionQueue::Next failed");
    }
    if (*got_tag != (void*)post_tag) {
        return errors::Internal("CompletionQueue::Next did not return the correct tag");
    }
    if (!ok) {
        return errors::Internal("CompletionQueue::Next did not return OK");
    }
    return Status::OK();
}

Status RuntimeGRPC::post_start(RuntimeStarter *starter, const uint32_t nn_id) {
    starter->request_.mutable_h_nn()->set_id(nn_id);
    starter->rpc_ = stub_->PrepareAsyncstart(
        &starter->context_, starter->request_, &starter->cq_);
    starter->post_tag_ = (int64_t)nn_id;
    starter->rpc_->StartCall();
    starter->rpc_->Finish(&starter->response_, &starter->status_, (void*)starter->post_tag_);
    return Status::OK();
}

Status RuntimeGRPC::wait_start(RuntimeStarter *starter) {
    void *got_tag;
    bool ok = false;
    TF_RETURN_IF_ERROR(wait_grpc_cq(&got_tag, &ok, &starter->cq_, starter->post_tag_));
    NRT_CHECK_RETURN("start", starter->status_, starter->response_);
    return Status::OK();
}

Status RuntimeGRPC::start_ping(const uint32_t nn_id) {
    // this function is only used as a hack to re-establish channel in case of grpc 14
    // and so intentionally returns OK as long as grpc status is ok
    nrt::start_request request;
    request.mutable_h_nn()->set_id(nn_id);
    nrt::start_response response;
    grpc::Status status = NRT_GRPC(stub_->start, request, &response);
    if (!status.ok()) {
        NRT_CHECK_RETURN("start", status, response);
    }
    return Status::OK();
}

Status RuntimeGRPC::setup_infer_post(RuntimeIO *runtime_io, int64_t post_tag) {
    runtime_io->rpc_infer_post_ = stub_->PrepareAsyncinfer_post(
        &runtime_io->context_, runtime_io->request_, &runtime_io->cq_);
    runtime_io->post_tag_ = post_tag;
    return Status::OK();
}

Status RuntimeGRPC::infer_post(RuntimeIO *runtime_io) {
    nrt::infer_post_response response;
    grpc::Status status = NRT_GRPC(stub_->infer_post, runtime_io->request_, &response);
    NRT_CHECK_RETURN("infer_post", status, response);
    runtime_io->cookie = response.cookie();
    return Status::OK();
}

Status RuntimeGRPC::post_infer_post(RuntimeIO *runtime_io) {
    if (nullptr == runtime_io->rpc_infer_post_) {
        return errors::Unavailable("runtime_io->rpc_infer_post_ is not initialized");
    }
    runtime_io->rpc_infer_post_->StartCall();
    runtime_io->rpc_infer_post_->Finish(&runtime_io->post_response_, &runtime_io->post_status_,
                                        (void*)runtime_io->post_tag_);
    return Status::OK();
}

Status RuntimeGRPC::wait_infer_post(RuntimeIO *runtime_io) {
    if (runtime_io->cookie != NRT_INVALID_COOKIE) {
        return Status::OK();
    }
    void *got_tag;
    bool ok = false;
    TF_RETURN_IF_ERROR(wait_grpc_cq(&got_tag, &ok, &runtime_io->cq_, runtime_io->post_tag_));
    NRT_CHECK_RETURN("infer_post", runtime_io->post_status_, runtime_io->post_response_);
    runtime_io->cookie = runtime_io->post_response_.cookie();
    return Status::OK();
}

Status RuntimeGRPC::infer_wait(RuntimeIO *runtime_io) {
    runtime_io->wait_request_.set_cookie(runtime_io->cookie);
    nrt::infer_response *response = &runtime_io->response_;

    // infer_wait
    grpc::Status status = NRT_GRPC(stub_->infer_wait, runtime_io->wait_request_, response);
    if (status.ok()) {
        // ignore inf/nan errors
        if (nrt::nerr::NERR_INFER_COMPLETED_WITH_NUM_ERR == response->status().code()) {
            response->mutable_status()->set_code(nrt::nerr::NERR_OK);
        }
    }
    NRT_CHECK_RETURN("infer_wait", status, *response);
    return Status::OK();
}

Status RuntimeGRPC::setup_infer(RuntimeIO *runtime_io, int64_t post_tag) {
    runtime_io->rpc_infer_ = stub_->PrepareAsyncinfer(
        &runtime_io->context_, runtime_io->request_, &runtime_io->cq_);
    runtime_io->post_tag_ = post_tag;
    return Status::OK();
}

Status RuntimeGRPC::post_infer(RuntimeIO *runtime_io) {
    if (nullptr == runtime_io->rpc_infer_) {
        return errors::Unavailable("runtime_io->rpc_infer_ is not initialized");
    }
    runtime_io->rpc_infer_->StartCall();
    runtime_io->rpc_infer_->Finish(&runtime_io->response_, &runtime_io->post_status_,
                                   (void*)runtime_io->post_tag_);
    return Status::OK();
}

Status RuntimeGRPC::wait_infer(RuntimeIO *runtime_io) {
    void *got_tag;
    bool ok = false;
    TF_RETURN_IF_ERROR(wait_grpc_cq(&got_tag, &ok, &runtime_io->cq_, runtime_io->post_tag_));
    nrt::infer_response *response = &runtime_io->response_;
    if (runtime_io->post_status_.ok()) {
        // ignore inf/nan errors
        if (nrt::nerr::NERR_INFER_COMPLETED_WITH_NUM_ERR == response->status().code()) {
            response->mutable_status()->set_code(nrt::nerr::NERR_OK);
        }
    }
    NRT_CHECK_RETURN("infer", runtime_io->post_status_, runtime_io->response_);
    return Status::OK();
}

Status RuntimeGRPC::stop(const uint32_t nn_id) {
    nrt::stop_request request;
    request.mutable_h_nn()->set_id(nn_id);
    nrt::stop_response response;
    grpc::Status status = NRT_GRPC(stub_->stop, request, &response);
    NRT_CHECK_RETURN("stop", status, response);
    return Status::OK();
}

Status RuntimeGRPC::post_stop(RuntimeStopper *stopper, const uint32_t nn_id) {
    stopper->request_.mutable_h_nn()->set_id(nn_id);
    stopper->rpc_ = stub_->PrepareAsyncstop(
        &stopper->context_, stopper->request_, &stopper->cq_);
    stopper->post_tag_ = (int64_t)nn_id;
    stopper->rpc_->StartCall();
    stopper->rpc_->Finish(&stopper->response_, &stopper->status_, (void*)stopper->post_tag_);
    return Status::OK();
}

Status RuntimeGRPC::wait_stop(RuntimeStopper *stopper) {
    void *got_tag;
    bool ok = false;
    TF_RETURN_IF_ERROR(wait_grpc_cq(&got_tag, &ok, &stopper->cq_, stopper->post_tag_));
    NRT_CHECK_RETURN("stop", stopper->status_, stopper->response_);
    return Status::OK();
}

Status RuntimeGRPC::unload(const uint32_t nn_id, bool from_global_state) {
    nrt::unload_request request;
    request.mutable_h_nn()->set_id(nn_id);
    nrt::unload_response response;
    grpc::Status status = NRT_GRPC(stub_->unload, request, &response);
    NRT_CHECK_RETURN("unload", status, response);
    return Status::OK();
}

Status RuntimeGRPC::destroy_eg(const uint32_t eg_id, bool from_global_state) {
    nrt::destroy_eg_request request;
    request.mutable_h_eg()->set_id(eg_id);
    nrt::destroy_eg_response response;
    grpc::Status status = NRT_GRPC(stub_->destroy_eg, request, &response);
    NRT_CHECK_RETURN("destroy_eg", status, response);
    return Status::OK();
}

Status RuntimeGRPC::shm_map(const std::string &path, const uint32_t mmap_prot, const uint64_t session_id) {
    nrt::shm_map_request request;
    request.set_path(path);
    request.set_mmap_prot(mmap_prot);
    if (RuntimeSession::INVALID_ID != session_id) {
        request.set_session_id(session_id);
    }
    nrt::shm_map_response response;
    grpc::Status status = NRT_GRPC(stub_->shm_map, request, &response);
    NRT_CHECK_RETURN("shm_map", status, response);
    return Status::OK();
}

Status RuntimeGRPC::shm_unmap(const std::string &path, const uint32_t mmap_prot) {
    nrt::shm_unmap_request request;
    request.set_path(path);
    request.set_mmap_prot(mmap_prot);
    nrt::shm_unmap_response response;
    grpc::Status status = NRT_GRPC(stub_->shm_unmap, request, &response);
    NRT_CHECK_RETURN("shm_unmap", status, response);
    return Status::OK();
}


RuntimeSession::RuntimeSession() {
}

Status RuntimeSession::initialize(const std::string& nrtd_address) {
    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    ch_args.SetMaxSendMessageSize(-1);
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        nrtd_address, grpc::InsecureChannelCredentials(), ch_args);
    if (!channel) {
        return errors::Unavailable(
            "cannot establish grpc channel to neuron-rtd server");
    }
    VLOG(1) << "session channel done";
    stub_ = nrt::nmgr_session_manager::NewStub(channel);
    if (!stub_) {
        return errors::Unavailable("cannot create stub");
    }

    // probe to determine whether session_mgr is implemented
    grpc::ClientContext context;
    std::shared_ptr<SessionReaderWriter> stream = stub_->session_monitor(&context);
    nrt::session_monitor_response probing_response;
    stream->Read(&probing_response);
    stream->WritesDone();
    grpc::Status status = stream->Finish();
    if (!status.ok() && grpc::StatusCode::UNAVAILABLE == status.error_code()) {
        return errors::Unavailable(
            "grpc server ", nrtd_address, " is unavailable. ",
            "Please check the status of neuron-rtd service by ",
            "`systemctl is-active neuron-rtd`. If it shows `inactive`, ",
            "please install the service by `sudo apt-get install aws-neuron-runtime`. ",
            "If `aws-neuron-runtime` is already installed, you may activate ",
            "neuron-rtd service by `sudo systemctl restart neuron-rtd`."
        );
    }
    if (status.error_code() == grpc::StatusCode::UNIMPLEMENTED) {
        id_ = INVALID_ID;
        return Status::OK();
    }

    // get session id from the actual stream
    context_ = std::make_shared<grpc::ClientContext>();
    stream_ = stub_->session_monitor(context_.get());
    nrt::session_monitor_response response;
    if (!stream_->Read(&response)) {
        return errors::Internal("error in reading session ID from neuron-rtd");
    }
    id_ = response.session_id();
    return Status::OK();
}

RuntimeSession::~RuntimeSession() {
    if (INVALID_ID != id_) {
        stream_->WritesDone();
        stream_->Finish();
        id_ = INVALID_ID;
    }
}


}  // namespace neuron
}  // namespace tensorflow

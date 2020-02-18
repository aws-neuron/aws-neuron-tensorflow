/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#include <grpcpp/grpcpp.h>
#include "nmgr.pb.h"
#include "nerr.pb.h"
#include "runtime_grpc.h"


namespace tensorflow {
namespace neuron {


Status RuntimeIO::setup(
        AttrList &input_names, const std::vector<const Tensor*> &input_tensors,
        AttrList &output_names, const std::vector<Tensor*> &output_tensors,
        const uint32_t nn_id, SharedMemory *shm) {
    shm_ = shm;
    for (auto idx = 0; idx < input_names.s_size(); ++idx) {
        nrt::infer_io *infer_io = request_.add_ifmap();
        infer_io->set_name(input_names.s(idx));
        StringPiece tensor_data(input_tensors[idx]->tensor_data());
        if (nullptr != shm_) {
            infer_io->mutable_buf_shm()->set_path(shm_->input_paths_[idx]);
            std::copy_n(tensor_data.data(), tensor_data.size(), shm_->input_ptrs_[idx]);
        } else {
            infer_io->set_buf(tensor_data.data(), tensor_data.size());
        }
    }
    if (nullptr != shm_) {
        for (int idx = 0; idx < output_names.s_size(); ++idx) {
            nrt::infer_io *infer_io = request_.add_shm_ofmap();
            infer_io->set_name(output_names.s(idx));
            infer_io->mutable_buf_shm()->set_path(shm_->output_paths_[idx]);
        }
        for (int idx = 0; idx < output_names.s_size(); ++idx) {
            nrt::infer_io *infer_io = wait_request_.add_shm_ofmap();
            infer_io->set_name(output_names.s(idx));
            infer_io->mutable_buf_shm()->set_path(shm_->output_paths_[idx]);
        }
    }
    request_.mutable_h_nn()->set_id(nn_id);
    output_names_ = &output_names;
    output_tensors_ = output_tensors;
    return Status::OK();
}

Status RuntimeIO::finish() {
    std::vector<StringPiece> raw_output_tensors;
    if (nullptr == shm_) {
        std::unordered_map<std::string, StringPiece> map_name_raw;
        for (const auto &infer_io : response_.ofmap()) {
            map_name_raw.emplace(infer_io.name(), infer_io.buf());
        }
        for (auto idx = 0; idx < output_names_->s_size(); ++idx) {
            if (map_name_raw.find(output_names_->s(idx)) == map_name_raw.end()) {
                return errors::Internal("tensor name", output_names_->s(idx),
                                        " not found in infer_response.ofmap()");
            }
            raw_output_tensors.push_back(map_name_raw[output_names_->s(idx)]);
        }
        for (auto idx = 0; idx < output_names_->s_size(); ++idx) {
            StringPiece out_tensor_raw = raw_output_tensors[idx];
            Tensor *out_tensor = output_tensors_[idx];
            TF_RETURN_WITH_CONTEXT_IF_ERROR(tensor_memcpy(out_tensor, out_tensor_raw),
                                            "tensor_memcpy failure on tensor name: ",
                                            output_names_->s(idx));
        }
        return Status::OK();
    }
    for (auto idx = 0; idx < output_names_->s_size(); ++idx) {
        StringPiece out_tensor_raw;
        if (nullptr != shm_) {
            out_tensor_raw = StringPiece(shm_->output_ptrs_[idx],
                                         shm_->output_sizes_[idx]);
        } else {
            out_tensor_raw = raw_output_tensors[idx];
        }
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            tensor_memcpy(output_tensors_[idx], out_tensor_raw),
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
                              const int num_cores_req) {
    nrt::create_eg_request request;
    if (num_cores_req >= 0) {
        request.set_nc_count((uint32_t)num_cores_req);
    }
    nrt::create_eg_response response;
    grpc::Status status = NRT_GRPC(stub_->create_eg, request, &response);
    if (!status.ok() && grpc::StatusCode::UNAVAILABLE == status.error_code()) {
        return errors::Unavailable(
            "grpc server ", nrtd_address_, " is unavailable. ",
            "Please check the status of neuron-rtd service by ",
            "`systemctl is-active neuron-rtd`. If it shows `inactive`, ",
            "please install the service by `sudo apt-get install aws-neuron-runtime`. ",
            "If `aws-neuron-runtime` is already installed, you may activate ",
            "neuron-rtd service by `sudo systemctl restart neuron-rtd`."
        );
    }
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

Status RuntimeGRPC::load(uint32_t *nn_id, const uint32_t eg_id,
                         const StringPiece &executable,
                         const uint32_t timeout, const uint32_t ninfer) {
    // load
    grpc::ClientContext context;
    nrt::load_response response;
    std::unique_ptr<grpc::ClientWriter<nrt::load_request> > writer(
        stub_->load(&context, &response));
    nrt::load_request request;

    #define WRITE_LOAD_REQUEST {                                                \
        if (!writer->Write(request)) {                                          \
            return errors::Internal("neuron-rtd load failure - broken stream"); \
        }                                                                       \
    }
    // eg_id
    request.mutable_h_eg()->set_id(eg_id);
    WRITE_LOAD_REQUEST;

    // neff_size
    size_t exec_total_size = executable.size();
    request.set_neff_size(exec_total_size);
    WRITE_LOAD_REQUEST;

    // model_params
    nrt::model_params *model_params = request.mutable_model_params();
    model_params->mutable_timeout()->set_data(timeout);
    model_params->mutable_ninfer()->set_data(ninfer);
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
        return errors::Internal("neuron-rtd load failure - broken stream");
    }
    grpc::Status status = writer->Finish();
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

Status RuntimeGRPC::setup_async_io(RuntimeIO *runtime_io, int64_t post_tag) {
    runtime_io->rpc_ = stub_->PrepareAsyncinfer_post(
        &runtime_io->context_, runtime_io->request_, &runtime_io->infer_post_cq_);
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
    if (nullptr == runtime_io->rpc_) {
        return errors::Internal("runtime_io->rpc_ is not initialized");
    }
    runtime_io->rpc_->StartCall();
    runtime_io->rpc_->Finish(&runtime_io->post_response_, &runtime_io->post_status_,
                             (void*)runtime_io->post_tag_);
    return Status::OK();
}

Status RuntimeGRPC::wait_infer_post(RuntimeIO *runtime_io) {
    if (runtime_io->cookie != NRT_INVALID_COOKIE) {
        return Status::OK();
    }
    void *got_tag;
    bool ok = false;
    if (!runtime_io->infer_post_cq_.Next(&got_tag, &ok)) {
        return errors::Internal("CompletionQueue::Next failed");
    }
    if (got_tag != (void*)runtime_io->post_tag_) {
        return errors::Internal("CompletionQueue::Next did not return the correct tag");
    }
    if (!ok) {
        return errors::Internal("CompletionQueue::Next did not return OK");
    }
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

Status RuntimeGRPC::stop(const uint32_t nn_id) {
    nrt::stop_request request;
    request.mutable_h_nn()->set_id(nn_id);
    nrt::stop_response response;
    grpc::Status status = NRT_GRPC(stub_->stop, request, &response);
    NRT_CHECK_RETURN("stop", status, response);
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

Status RuntimeGRPC::shm_map(const std::string &path, const uint32_t mmap_prot) {
    nrt::shm_map_request request;
    request.set_path(path);
    request.set_mmap_prot(mmap_prot);
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


}  // namespace neuron
}  // namespace tensorflow

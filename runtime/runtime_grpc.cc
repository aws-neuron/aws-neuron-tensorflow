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

#include "runtime_grpc.h"
#include <grpcpp/grpcpp.h>
#include "nmgr.pb.h"
#include "version.h"

namespace tensorflow {
namespace neuron {

Status RuntimeIO::setup(AttrList& input_names, AttrList& output_names,
                        const std::vector<Tensor*>& output_tensors,
                        const std::vector<Tensor*>& output_shm_tensors,
                        const uint32_t nn_id, bool use_shm,
                        const std::vector<StringPiece>& input_paths,
                        const std::vector<StringPiece>& output_paths,
                        thread::ThreadPool* thread_pool) {
  CHECK_SIZES_MATCH(output_names.s_size(), output_tensors.size());
  CHECK_SIZES_MATCH(output_tensors.size(), output_shm_tensors.size());
  thread_pool_ = thread_pool;
  use_shm_ = use_shm;
  output_shm_tensors_ = output_shm_tensors;
  for (auto idx = 0; idx < input_names.s_size(); ++idx) {
    nrt::infer_io* infer_io = request_.add_ifmap();
    infer_io->set_name(input_names.s(idx));
    if (TF_PREDICT_TRUE(use_shm_)) {
      StringPiece path = input_paths.at(idx);
      infer_io->mutable_buf_shm()->set_path(path.data(), path.size());
    }
  }
  if (TF_PREDICT_TRUE(use_shm_)) {
    for (int idx = 0; idx < output_names.s_size(); ++idx) {
      StringPiece path = output_paths.at(idx);
      nrt::infer_io* infer_io = request_.add_shm_ofmap();
      infer_io->set_name(output_names.s(idx));
      infer_io->mutable_buf_shm()->set_path(path.data(), path.size());
      nrt::infer_io* infer_io_wait = wait_request_.add_shm_ofmap();
      infer_io_wait->set_name(output_names.s(idx));
      infer_io_wait->mutable_buf_shm()->set_path(path.data(), path.size());
    }
  }
  request_.mutable_h_nn()->set_id(nn_id);
  output_names_ = &output_names;
  output_tensors_ = output_tensors;
  return Status::OK();
}

Status RuntimeIO::copy_input_tensors(
    const std::vector<const Tensor*>& input_tensors) {
  CHECK_SIZES_MATCH(request_.ifmap_size(), input_tensors.size());
  if (TF_PREDICT_FALSE(!use_shm_)) {
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      nrt::infer_io* infer_io = request_.mutable_ifmap(idx);
      StringPiece tensor_data(input_tensors[idx]->tensor_data());
      infer_io->set_buf(tensor_data.data(), tensor_data.size());
    }
  }
  return Status::OK();
}

Status RuntimeIO::finish() {
  if (TF_PREDICT_FALSE(!use_shm_)) {
    std::vector<StringPiece> raw_output_tensors;
    std::unordered_map<std::string, StringPiece> map_name_raw;
    for (const auto& infer_io : response_.ofmap()) {
      map_name_raw.emplace(infer_io.name(), infer_io.buf());
    }
    for (auto idx = 0; idx < output_names_->s_size(); ++idx) {
      if (map_name_raw.find(output_names_->s(idx)) == map_name_raw.end()) {
        return errors::NotFound("tensor name", output_names_->s(idx),
                                " not found in infer_response.ofmap()");
      }
      raw_output_tensors.push_back(map_name_raw[output_names_->s(idx)]);
    }
    for (auto idx = 0; idx < output_names_->s_size(); ++idx) {
      Tensor* out_tensor = output_tensors_[idx];
      StringPiece out_tensor_raw = raw_output_tensors.at(idx);
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          tensor_memcpy(out_tensor, out_tensor_raw, thread_pool_),
          "tensor_memcpy failure on tensor name: ", output_names_->s(idx));
    }
  } else {
    for (auto idx = 0; idx < output_names_->s_size(); ++idx) {
      Tensor* out_tensor = output_tensors_[idx];
      const Tensor& shm_tensor = *output_shm_tensors_.at(idx);
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          tensor_copy(out_tensor, shm_tensor, thread_pool_),
          "tensor_copy failure on tensor name: ", output_names_->s(idx));
    }
  }
  return Status::OK();
}

Status RuntimeGRPC::initialize(const std::string& nrtd_address) {
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

Status RuntimeGRPC::create_eg(uint32_t* eg_id, uint32_t* num_cores,
                              const int num_cores_req,
                              const uint64_t session_id) {
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
        "YOUR OTHER PROCESSES RUNNING ON MACHINE LEARNING ACCELERATORS!");
  }
  NRT_CHECK_RETURN("create_eg", status, response);
  *eg_id = response.h_eg().id();
  *num_cores = response.nc_count();
  return Status::OK();
}

Status RuntimeGRPC::load(uint32_t* nn_id, const uint32_t eg_id,
                         const StringPiece& executable, const uint32_t timeout,
                         const uint32_t ninfer, const bool profile_enabled,
                         const uint64_t session_id) {
  // load
  grpc::ClientContext context;
  nrt::load_response response;
  std::unique_ptr<grpc::ClientWriter<nrt::load_request> > writer(
      stub_->load(&context, &response));
  nrt::load_request request;

#define WRITE_LOAD_REQUEST                                                \
  {                                                                       \
    if (!writer->Write(request)) {                                        \
      return errors::DataLoss("neuron-rtd load failure - broken stream"); \
    }                                                                     \
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
  nrt::model_params* model_params = request.mutable_model_params();
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
    bool unsupp(nrt::nerr::NERR_UNSUPPORTED_VERSION ==
                response.status().code());
    bool invalid(nrt::nerr::NERR_INVALID == response.status().code());
    bool unsupp_parsed(response.status().details().find(
                           "NEFF version mismatch") != std::string::npos);
    if (unsupp ||
        (invalid && (unsupp_parsed || "" == response.status().details()))) {
      return errors::InvalidArgument(
          "Neuron runtime could not load the compiled SavedModel provided "
          "because it is "
          "compiled with a neuron-cc that is not compatible with the runtime "
          "version used. "
          "Please make sure you upgrade the Neuron runtime version used. "
          "(neuron-rtd error code: ",
          response.status().code(), ", details: ", response.status().details(),
          ")");
    }
  }
  NRT_CHECK_RETURN("load", status, response);
  *nn_id = response.h_nn().id();
  return Status::OK();
}

static Status wait_grpc_cq(grpc::CompletionQueue* cq, const int64_t post_tag) {
  void* got_tag;
  bool ok = false;
  if (TF_PREDICT_FALSE(!cq->Next(&got_tag, &ok))) {
    return errors::Internal("CompletionQueue::Next failed");
  }
  if (TF_PREDICT_FALSE(got_tag != (void*)post_tag)) {
    return errors::Internal(
        "CompletionQueue::Next did not return the correct tag");
  }
  if (TF_PREDICT_FALSE(!ok)) {
    return errors::Internal("CompletionQueue::Next did not return OK");
  }
  return Status::OK();
}

Status RuntimeGRPC::post_start(RuntimeStarter* starter, const uint32_t nn_id) {
  starter->request_.mutable_h_nn()->set_id(nn_id);
  starter->rpc_ =
      stub_->Asyncstart(&starter->context_, starter->request_, &starter->cq_);
  starter->post_tag_ = (int64_t)nn_id;
  starter->rpc_->Finish(&starter->response_, &starter->status_,
                        (void*)starter->post_tag_);
  return Status::OK();
}

Status RuntimeGRPC::wait_start(RuntimeStarter* starter) {
  TF_RETURN_IF_ERROR(wait_grpc_cq(&starter->cq_, starter->post_tag_));
  NRT_CHECK_RETURN("start", starter->status_, starter->response_);
  return Status::OK();
}

Status RuntimeGRPC::infer_post(RuntimeIO* io) {
  io->post_rpc_ =
      stub_->Asyncinfer_post(&io->post_context_, io->request_, &io->cq_);
  int64_t post_tag = 1;
  io->post_rpc_->Finish(&io->post_response_, &io->post_status_,
                        (void*)post_tag);
  return wait_grpc_cq(&io->cq_, post_tag);
}

Status RuntimeGRPC::infer_wait(RuntimeIO* io) {
  NRT_CHECK_RETURN("infer_post", io->post_status_, io->post_response_);
  io->wait_request_.set_cookie(io->post_response_.cookie());
  int64_t wait_tag = 2;
  io->wait_rpc_ =
      stub_->Asyncinfer_wait(&io->wait_context_, io->wait_request_, &io->cq_);
  io->wait_rpc_->Finish(&io->response_, &io->wait_status_, (void*)wait_tag);
  TF_RETURN_IF_ERROR(wait_grpc_cq(&io->cq_, wait_tag));
  if (TF_PREDICT_TRUE(io->wait_status_.ok())) {
    // ignore inf/nan errors
    const int code = io->response_.status().code();
    if (TF_PREDICT_FALSE(nrt::nerr::NERR_INFER_COMPLETED_WITH_NUM_ERR ==
                         code)) {
      io->response_.mutable_status()->set_code(nrt::nerr::NERR_OK);
    }
  }
  NRT_CHECK_RETURN("infer_wait", io->wait_status_, io->response_);
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

Status RuntimeGRPC::post_stop(RuntimeStopper* stopper, const uint32_t nn_id) {
  stopper->request_.mutable_h_nn()->set_id(nn_id);
  stopper->rpc_ =
      stub_->Asyncstop(&stopper->context_, stopper->request_, &stopper->cq_);
  stopper->post_tag_ = (int64_t)nn_id;
  stopper->rpc_->Finish(&stopper->response_, &stopper->status_,
                        (void*)stopper->post_tag_);
  return Status::OK();
}

Status RuntimeGRPC::wait_stop(RuntimeStopper* stopper) {
  TF_RETURN_IF_ERROR(wait_grpc_cq(&stopper->cq_, stopper->post_tag_));
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

Status RuntimeGRPC::shm_map(const std::string& path, const uint32_t mmap_prot,
                            const uint64_t session_id) {
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

Status RuntimeGRPC::shm_unmap(const std::string& path,
                              const uint32_t mmap_prot) {
  nrt::shm_unmap_request request;
  request.set_path(path);
  request.set_mmap_prot(mmap_prot);
  nrt::shm_unmap_response response;
  grpc::Status status = NRT_GRPC(stub_->shm_unmap, request, &response);
  NRT_CHECK_RETURN("shm_unmap", status, response);
  return Status::OK();
}

RuntimeSession::RuntimeSession() {}

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
  std::shared_ptr<SessionReaderWriter> stream =
      stub_->session_monitor(&context);
  nrt::session_monitor_response probing_response;
  stream->Read(&probing_response);
  stream->WritesDone();
  grpc::Status status = stream->Finish();
  if (!status.ok() && grpc::StatusCode::UNAVAILABLE == status.error_code()) {
    return errors::Unavailable(
        "grpc server ", nrtd_address, " is unavailable. ",
        "Please check the status of neuron-rtd service by ",
        "`systemctl is-active neuron-rtd`. If it shows `inactive`, ",
        "please install the service by `sudo apt-get install "
        "aws-neuron-runtime`. ",
        "If `aws-neuron-runtime` is already installed, you may activate ",
        "neuron-rtd service by `sudo systemctl restart neuron-rtd`.");
  }
  if (status.error_code() == grpc::StatusCode::UNIMPLEMENTED) {
    id_ = INVALID_ID;
    return Status::OK();
  }

  // get session id from the actual stream
  context_ = std::make_shared<grpc::ClientContext>();
  stream_ = stub_->session_monitor(context_.get());
  nrt::session_monitor_request request;
  request.set_framework_name("TENSORFLOW");
  nrt::version* framework_version = request.mutable_framework_version();
  framework_version->set_full_version(TF_VERSION_STRING);
  framework_version->set_major_num(TF_MAJOR_VERSION);
  framework_version->set_minor_num(TF_MINOR_VERSION);
  nrt::version* fal_version = request.mutable_fal_version();
  fal_version->set_full_version(TFN_VERSION_STRING);
  fal_version->set_major_num(TFN_MAJOR_VERSION);
  fal_version->set_minor_num(TFN_MINOR_VERSION);
  if (!stream_->Write(request)) {
    return errors::Internal("error in writing session request to neuron-rtd");
  }
  nrt::session_monitor_response response;
  if (!stream_->Read(&response)) {
    return errors::Internal("error in reading session ID from neuron-rtd");
  }
  VLOG(1) << "passed tensorflow version " << TF_VERSION_STRING << " to runtime";
  VLOG(1) << "passed tensorflow-neuron version " << TFN_VERSION_STRING
          << " to runtime";
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

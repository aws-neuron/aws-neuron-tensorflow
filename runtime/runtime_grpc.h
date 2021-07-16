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
#include "nerr.pb.h"
#include "nmgr_service.grpc.pb.h"
#include "nmgr_session_service.grpc.pb.h"
#include "tensor_util.h"

namespace tensorflow {
namespace neuron {

#define ASYNC_GRPC_INVALID_TAG -1

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
  int64_t post_tag_ = ASYNC_GRPC_INVALID_TAG;

 private:
  TFN_DISALLOW_COPY_MOVE_ASSIGN(RuntimeSwitcher);
};
typedef RuntimeSwitcher<nrt::start_request, nrt::start_response> RuntimeStarter;
typedef RuntimeSwitcher<nrt::stop_request, nrt::stop_response> RuntimeStopper;

class RuntimeIO {
 public:
  RuntimeIO() {}
  Status setup(AttrList& input_names, AttrList& output_names,
               const uint32_t nn_id, bool use_shm,
               const std::vector<StringPiece>& input_paths,
               const std::vector<StringPiece>& output_paths);
  bool use_shm() { return use_shm_; }
  Status copy_input_tensors(const std::vector<Tensor>& input_tensors);
  void set_nn_id(const uint32_t nn_id) {
    request_.mutable_h_nn()->set_id(nn_id);
  }
  uint32_t get_nn_id() { return request_.mutable_h_nn()->id(); }
  Status finish(std::vector<Tensor*>* output_tensors,
                const std::vector<Tensor>& output_shm_tensors,
                thread::ThreadPool* thread_pool);

 private:
  friend class RuntimeGRPC;
  grpc::ClientContext post_context_;
  grpc::CompletionQueue cq_;
  std::unique_ptr<grpc::ClientAsyncResponseReader<nrt::infer_post_response> >
      post_rpc_ = nullptr;
  nrt::infer_request request_;
  nrt::infer_post_response post_response_;
  grpc::Status post_status_;
  grpc::ClientContext wait_context_;
  std::unique_ptr<grpc::ClientAsyncResponseReader<nrt::infer_response> >
      wait_rpc_ = nullptr;
  nrt::infer_wait_request wait_request_;
  nrt::infer_response response_;
  grpc::Status wait_status_;
  AttrList* output_names_;
  bool use_shm_ = false;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(RuntimeIO);
};

class RuntimeGRPC {
 public:
  RuntimeGRPC() {}
  Status initialize(const std::string& nrtd_address);
  Status create_eg(uint32_t* eg_id, uint32_t* num_cores,
                   const int num_cores_req, const uint64_t session_id);
  Status load(uint32_t* nn_id, const uint32_t eg_id,
              const StringPiece& executable, const uint32_t timeout,
              const uint32_t ninfer, const bool profile_enabled,
              const uint64_t session_id);
  Status start_ping(const uint32_t nn_id);
  Status post_start(RuntimeStarter* starter, const uint32_t nn_id);
  Status wait_start(RuntimeStarter* starter);
  Status infer_post(RuntimeIO* runtime_io);
  Status infer_wait(RuntimeIO* runtime_io);
  Status stop(const uint32_t nn_id);
  Status post_stop(RuntimeStopper* stopper, const uint32_t nn_id);
  Status wait_stop(RuntimeStopper* stopper);
  Status unload(const uint32_t nn_id, bool from_global_state = false);
  Status destroy_eg(const uint32_t eg_id, bool from_global_state = false);
  Status shm_map(const std::string& path, const uint32_t mmap_prot,
                 const uint64_t session_id);
  Status shm_unmap(const std::string& path, const uint32_t mmap_prot);

 private:
  std::unique_ptr<nrt::nmgr_v1::Stub> stub_;
  // some reasonable number of bytes
  static const size_t EXEC_MAX_CHUNK_SIZE = 1024 * 1024;
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
  typedef grpc::ClientReaderWriter<nrt::session_monitor_request,
                                   nrt::session_monitor_response>
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

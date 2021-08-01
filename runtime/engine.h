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

#ifndef TENSORFLOW_NEURON_RUNTIME_ENGINE_H_
#define TENSORFLOW_NEURON_RUNTIME_ENGINE_H_

#include <queue>
#include "profiler.h"
#include "runtime_grpc.h"
#include "semaphore.h"
#include "shared_memory.h"
#include "tensor_util.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace neuron {

typedef std::queue<xla::Semaphore::ScopedReservation> SemResQueue;

class NeuronEngine {
 public:
  NeuronEngine() {}
  Status initialize(const std::string& nrtd_address, const int num_cores_req,
                    const int num_dup, std::shared_ptr<RuntimeSession> session);
  Status load(uint32_t* nn_id, const StringPiece& executable,
              const uint32_t timeout, const uint32_t ninfer,
              const bool profile_enabled);
  Status start_ping();
  Status infer(RuntimeIO* runtime_io);
  Status infer_with_profiling(RuntimeIO* runtime_io,
                              ProfilerInterface* profile);
  void unload(const uint32_t nn_id);
  void clear(bool from_global_state = false);
  size_t num_executable() { return nn_id_to_all_nn_ids_.size(); };
  uint32_t num_cores() { return num_cores_; };
  std::shared_ptr<RuntimeSession> get_session() { return session_; }
  thread::ThreadPool* get_thread_pool() { return thread_pool_.get(); }

 private:
  Status start_model_unsafe(const uint32_t nn_id);
  bool is_busy();
  bool running(uint32_t nn_id);
  void set_running(uint32_t nn_id);
  uint32_t nn_get_current_running();
  Status get_active(uint32_t* active_nn_id,
                    std::shared_ptr<xla::Semaphore>* sem, const uint32_t nn_id);
  tensorflow::mutex mutex_eg_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  bool closed_ = false;
  RuntimeGRPC runtime_;
  uint64_t session_id_ = RuntimeSession::INVALID_ID;
  std::shared_ptr<RuntimeSession> session_ = nullptr;
  std::vector<uint32_t> vec_eg_id_;
  uint64 last_active_timestamp_ = 0;
  uint32_t running_nn_id_;
  uint32_t num_cores_ = 0;
  std::string nrtd_address_ = "";
  std::unordered_map<uint32_t, std::vector<uint32_t> > nn_id_to_all_nn_ids_;
  std::unordered_map<uint32_t, size_t> nn_id_to_active_idx_;
  std::unordered_map<uint32_t, std::vector<std::shared_ptr<xla::Semaphore> > >
      nn_id_to_sems_;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronEngine);
};

class NeuronEngineManager {
 public:
  static NeuronEngineManager& GetNeuronEngineManager();
  SharedMemoryAllocator* get_shm_allocator() { return shm_allocator_.get(); }
  Status apply_for_engine(NeuronEngine** engine,
                          const std::string& session_handle,
                          const int64_t opt_engine_size,
                          const int64_t max_num_duplicates,
                          const int64_t engine_index = -1);
  void clear_if_empty();
  void clear_from_global_state();
  static const int64 MAX_NUM_CORES = 64;
  static const int64 MIN_NUM_CORES = 0;

 private:
  NeuronEngineManager();
  ~NeuronEngineManager();
  std::vector<std::pair<int, int>> get_default_device_spec(
      const int64_t opt_engine_size, const int64_t max_num_duplicates);
  Status init_engines(const std::vector<std::pair<int, int>>& device_specs);
  Status initialize(const int64_t opt_engine_size,
                    const int64_t max_num_duplicates);
  void clear();
  tensorflow::mutex global_mutex_;
  static const int64_t ONE_DEVICE_NUM_CORES = 4;
  static const int DEFAULT_NUM_CORES = -65536;  // any number < -MAX_NUM_CORES
  std::string nrtd_address_;
  std::shared_ptr<RuntimeSession> session_ = nullptr;
  std::shared_ptr<SharedMemoryAllocator> shm_allocator_ = nullptr;
  std::array<NeuronEngine, MAX_NUM_CORES> engine_array_;
  std::unordered_map<std::string, size_t> session_handle_to_engine_index_;
  size_t engine_index_ = 0;
  size_t num_engines_ = 0;
  Status runtime_status_ = errors::InvalidArgument("Uninitialized");
  bool ready_ = false;
  TFN_DISALLOW_COPY_MOVE_ASSIGN(NeuronEngineManager);
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_ENGINE_H_

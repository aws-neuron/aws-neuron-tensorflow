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

#include "device.h"
#include "env.h"
#include "macros.h"

namespace tensorflow {
namespace neuron {

static std::string remove_pattern(std::string data,
                                  const std::string& pattern) {
  size_t string_length = data.size();
  size_t pos = 0;
  for (size_t idx = 0; idx < string_length; ++idx) {
    pos = data.find(pattern, pos);
    if (std::string::npos == pos) {
      break;
    }
    data.replace(pos, pattern.size(), "");
  }
  return data;
}

NeuronDeviceManager::NeuronDeviceManager() {
  tensorflow::mutex_lock lock(global_mutex_);
  // append /opt/aws/neuron/bin to PATH
  std::string env_path = env_get("PATH", "");
  setenv("PATH", (env_path + ":/opt/aws/neuron/bin").c_str(), 1);

  // neuron-rtd address
  nrtd_address_ = env_get("NEURON_RTD_ADDRESS", "unix:/run/neuron.sock");

  // runtime session
  session_ = std::make_shared<RuntimeSession>();
  runtime_status_ = session_->initialize(nrtd_address_);

  // shared memory allocator
  shm_buf_mgr_ = std::make_shared<SharedMemoryBufferManager>();
  if (runtime_status_.ok()) {
    shm_buf_mgr_->initialize(session_->get_id(), nrtd_address_);
  }
}

NeuronDeviceManager::~NeuronDeviceManager() {
  tensorflow::mutex_lock lock(global_mutex_);
  clear_from_global_state();
}

Status NeuronDeviceManager::initialize(const int64_t opt_device_size,
                                       const int64_t max_num_duplicates) {
  TF_RETURN_IF_ERROR(runtime_status_);

  // get number of neuron cores from comma-separated list of integers
  std::string neuron_device_sizes_raw = env_get("NEURONCORE_GROUP_SIZES", "");
  if (neuron_device_sizes_raw.empty()) {
    TF_RETURN_IF_ERROR(
        init_default_device(opt_device_size, max_num_duplicates));
  } else {
    // remove [ and ]
    std::string neuron_device_sizes =
        remove_pattern(neuron_device_sizes_raw, "[");
    neuron_device_sizes = remove_pattern(neuron_device_sizes, "]");

    std::vector<int> num_cores_req_vector;
    std::vector<int> num_dup_vector;
    std::stringstream neuron_device_sizes_stream(neuron_device_sizes);
    for (size_t idx = 0; idx < MAX_NUM_CORES; ++idx) {
      if (!neuron_device_sizes_stream.good()) {
        break;
      }
      std::string device_spec;
      std::getline(neuron_device_sizes_stream, device_spec, ',');
      if (device_spec.empty()) {
        continue;
      }
      int num_dup = 1;
      if (device_spec.find("x") != std::string::npos) {
        size_t delim_pos = device_spec.find("x");
        num_dup = stoi_no_throw(device_spec.substr(0, delim_pos));
        device_spec = device_spec.substr(delim_pos + 1, std::string::npos);
      }
      int num_cores_req = stoi_no_throw(device_spec);
      if (num_cores_req < 0 || num_cores_req > MAX_NUM_CORES || num_dup <= 0 ||
          num_dup > MAX_NUM_CORES) {
        LOG(WARNING) << "NEURONCORE_GROUP_SIZES=" << neuron_device_sizes_raw
                     << " looks ill-formatted. Falling back to initializing"
                     << " a default NeuronCore Group.";
        num_cores_req_vector.clear();
        num_dup_vector.clear();
        break;
      }
      num_cores_req_vector.push_back(num_cores_req);
      num_dup_vector.push_back(num_dup);
    }
    if (num_cores_req_vector.empty()) {
      TF_RETURN_IF_ERROR(
          init_default_device(opt_device_size, max_num_duplicates));
    } else {
      TF_RETURN_IF_ERROR(init_devices(num_cores_req_vector, num_dup_vector));
    }
  }
  ready_ = true;
  return Status::OK();
}

Status NeuronDeviceManager::init_devices(
    const std::vector<int>& num_cores_req_vector,
    const std::vector<int>& num_dup_vector) {
  Status status =
      errors::ResourceExhausted("No NeuronCore Group can be initialized.");
  for (size_t idx = 0; idx < num_cores_req_vector.size(); ++idx) {
    int num_cores_req = num_cores_req_vector[idx];
    int num_dup;
    if (num_dup_vector.size() == num_cores_req_vector.size()) {
      num_dup = num_dup_vector[idx];
    } else {
      num_dup = 1;
    }
    if (nullptr == session_) {
      return errors::Internal("neuron runtime session is not initialized");
    }
    status = device_array_[idx].initialize(nrtd_address_, num_cores_req,
                                           num_dup, session_, shm_buf_mgr_);
    if (!status.ok()) {
      if (status.code() != tensorflow::error::Code::ABORTED) {
        LOG(WARNING) << "Cannot initialize NeuronCore Group with "
                     << num_cores_req << " cores; stopping initialization.";
      }
      break;
    }
    ++num_devices_;
    VLOG(1) << "successfully initialized NeuronCore Group of size "
            << num_cores_req;
  }
  if (0 == num_devices_) {
    return status;
  }
  return Status::OK();
}

Status NeuronDeviceManager::init_default_device(
    const int64_t opt_device_size, const int64_t max_num_duplicates) {
  std::vector<int> num_cores_req_vector = {DEFAULT_NUM_CORES};
  std::vector<int> num_dup_vector({1});
  if (0 <= opt_device_size && opt_device_size <= 64) {
    // get one full Inferentia by default
    if (opt_device_size == 1) {
      if (4 == max_num_duplicates) {
        num_cores_req_vector = {1};
        num_dup_vector = {4};
      } else if (3 == max_num_duplicates) {
        num_cores_req_vector = {1};
        num_dup_vector = {3};
      } else if (2 == max_num_duplicates) {
        num_cores_req_vector = {1, 1};
        num_dup_vector = {2, 2};
      } else {
        num_cores_req_vector = {1, 1, 1, 1};
        num_dup_vector = {};
      }
    } else if (opt_device_size == 2) {
      if (2 == max_num_duplicates) {
        num_cores_req_vector = {2};
        num_dup_vector = {2};
      } else {
        num_cores_req_vector = {2, 2};
        num_dup_vector = {};
      }
    }
  }
  return init_devices(num_cores_req_vector, num_dup_vector);
}

void NeuronDeviceManager::clear_if_empty() {
  tensorflow::mutex_lock lock(global_mutex_);
  bool empty = true;
  for (size_t idx = 0; idx < num_devices_; ++idx) {
    if (0 != device_array_[idx].num_executable()) {
      empty = false;
    }
  }
  if (empty) {
    clear();
  }
}

void NeuronDeviceManager::clear() {
  for (size_t idx = 0; idx < num_devices_; ++idx) {
    device_array_[idx].clear();
  }
  session_handle_to_device_index_.clear();
  num_devices_ = 0;
  device_index_ = 0;
  ready_ = false;
  VLOG(1) << "NeuronDeviceManager is cleared";
}

void NeuronDeviceManager::clear_from_global_state() {
  for (size_t idx = 0; idx < num_devices_; ++idx) {
    device_array_[idx].clear(true);
  }
  session_handle_to_device_index_.clear();
  num_devices_ = 0;
  device_index_ = 0;
  ready_ = false;
  VLOG(1) << "NeuronDeviceManager is cleared from global state";
}

NeuronDeviceManager& NeuronDeviceManager::GetNeuronDeviceManager() {
  static NeuronDeviceManager mgr;
  return mgr;
}

Status NeuronDeviceManager::apply_for_device(NeuronDevice** device,
                                             const std::string& session_handle,
                                             const int64_t opt_device_size,
                                             const int64_t max_num_duplicates,
                                             const int64_t device_index) {
  tensorflow::mutex_lock lock(global_mutex_);
  if (!ready_) {
    TF_RETURN_IF_ERROR(initialize(opt_device_size, max_num_duplicates));
  }

  // a particular device_index is requested by the client
  if (0 <= device_index && device_index < (int64_t)num_devices_) {
    *device = &device_array_[device_index];
    return Status::OK();
  }

  // if seeing a NEFF that is in the same session as a previously seen NEFF is
  // in, then prefer giving them the same device
  if (session_handle_to_device_index_.count(session_handle)) {
    *device = &device_array_[session_handle_to_device_index_[session_handle]];
    return Status::OK();
  }

  // otherwise get the next device and round-robin
  *device = &device_array_[device_index_];
  session_handle_to_device_index_[session_handle] = device_index_;
  ++device_index_;
  if (device_index_ >= num_devices_) {
    device_index_ = 0;
  }
  return Status::OK();
}

Status NeuronDevice::initialize(
    const std::string& nrtd_address,
    const int num_cores_req, const int num_dup,
    std::shared_ptr<RuntimeSession> session,
    std::shared_ptr<SharedMemoryBufferManager> shm_buf_mgr) {
  tensorflow::mutex_lock lock(mutex_eg_);
  if (closed_) {
    return errors::Aborted("neuron_device is closed");
  }
  nrtd_address_ = nrtd_address;
  TF_RETURN_IF_ERROR(runtime_.initialize(nrtd_address_));
  if (nullptr == session) {
    return errors::Internal("neuron runtime session is not initialized");
  }
  session_ = session;
  uint64_t session_id = session->get_id();
  if (RuntimeSession::INVALID_ID != session_id) {
    session_id_ = session_id;
  }
  if (num_dup == 1) {
    uint32_t eg_id = NRT_INVALID_EG_ID;
    TF_RETURN_IF_ERROR(
        runtime_.create_eg(&eg_id, &num_cores_, num_cores_req, session_id));
    vec_eg_id_.push_back(eg_id);
  } else {
    // setup device to duplicate models automatically
    for (int idx = 0; idx < num_dup; ++idx) {
      uint32_t eg_id = NRT_INVALID_EG_ID;
      uint32_t num_cores = 0;
      TF_RETURN_IF_ERROR(
          runtime_.create_eg(&eg_id, &num_cores, num_cores_req, session_id));
      vec_eg_id_.push_back(eg_id);
      num_cores_ = num_cores_req;
    }
  }
  running_nn_id_ = NRT_INVALID_NN_ID;
  shm_buf_mgr_ = shm_buf_mgr;
  return Status::OK();
}

Status NeuronDevice::load(uint32_t* nn_id, const StringPiece& executable,
                          const uint32_t timeout, const uint32_t ninfer,
                          const bool profile_enabled) {
  tensorflow::mutex_lock lock(mutex_eg_);
  if (closed_) {
    return errors::Aborted("neuron_device is closed");
  }
  uint32_t first_nn_id = NRT_INVALID_NN_ID;
  std::vector<uint32_t> all_nn_ids;
  if (vec_eg_id_.size() == 1) {
    TF_RETURN_IF_ERROR(runtime_.load(&first_nn_id, vec_eg_id_[0], executable,
                                     timeout, ninfer, profile_enabled,
                                     session_id_));
    all_nn_ids.push_back(first_nn_id);
  } else if (vec_eg_id_.size() > 1) {
    Status status;
    for (const uint32_t eg_id : vec_eg_id_) {
      uint32_t this_nn_id = NRT_INVALID_NN_ID;
      status = runtime_.load(&this_nn_id, eg_id, executable, timeout, ninfer,
                             profile_enabled, session_id_);
      if (!status.ok()) {
        LOG(WARNING) << "stop duplicating nn " << first_nn_id
                     << " due to error " << status.error_message();
        break;
      }
      if (all_nn_ids.size() == 0) {
        TF_RETURN_IF_ERROR(status);
        first_nn_id = this_nn_id;
      } else {
        VLOG(1) << "duplicated " << first_nn_id << " as " << this_nn_id;
      }
      all_nn_ids.push_back(this_nn_id);
    }
    if (all_nn_ids.size() == 0) {
      return status;
    }
  } else {
    return errors::Unavailable("NeuronDevice is uninitialized");
  }
  if (nn_id_to_all_nn_ids_.count(first_nn_id)) {
    for (const uint32_t nid : all_nn_ids) {
      TF_LOG_IF_ERROR(runtime_.unload(nid));
    }
    return errors::AlreadyExists("nn ", first_nn_id, " is already mapped");
  }
  nn_id_to_all_nn_ids_[first_nn_id] = all_nn_ids;
  nn_id_to_active_idx_[first_nn_id] = 0;
  std::vector<std::shared_ptr<xla::Semaphore> >& sems =
      nn_id_to_sems_[first_nn_id];
  for (const auto& nn_id : all_nn_ids) {
    VLOG(1) << "model " << nn_id << " infer semaphore capacity " << ninfer;
    sems.push_back(std::make_shared<xla::Semaphore>(ninfer));
  }
  *nn_id = first_nn_id;
  VLOG(1) << "successfully loaded " << first_nn_id;
  return Status::OK();
}

void NeuronDevice::unload(const uint32_t nn_id) {
  tensorflow::mutex_lock lock(mutex_eg_);
  if (closed_) {
    return;
  }
  if (!nn_id_to_all_nn_ids_.count(nn_id)) {
    VLOG(1) << "model " << nn_id << " is not loaded";
    return;
  }
  // stop
  if (running(nn_id)) {
    // stop all models
    for (const uint32_t nid : nn_id_to_all_nn_ids_[nn_id]) {
      TF_LOG_IF_ERROR(runtime_.stop(nid));
    }
    set_running(NRT_INVALID_NN_ID);
  }

  // unload all models
  for (const uint32_t nid : nn_id_to_all_nn_ids_[nn_id]) {
    TF_LOG_IF_ERROR(runtime_.unload(nid));
  }
  nn_id_to_all_nn_ids_.erase(nn_id);
  VLOG(1) << "unload: number of NEFFs: " << num_executable();
}

Status NeuronDevice::setup_scoped_runtime_io(
    ScopedRuntimeIO* scoped_io, AttrList& input_names,
    const std::vector<const Tensor*>& input_tensors, AttrList& output_names,
    const std::vector<size_t>& output_tensor_sizes,
    const std::vector<Tensor*>& output_tensors, const uint32_t nn_id,
    thread::ThreadPool* thread_pool) {
  if (nullptr == scoped_io) {
    return errors::Internal("bad ScopedRuntimeIO pointer");
  }
  return scoped_io->setup(input_names, input_tensors, output_names,
                          output_tensor_sizes, output_tensors, nn_id,
                          thread_pool, shm_buf_mgr_);
}

Status NeuronDevice::infer(ScopedRuntimeIO* scoped_io) {
  RuntimeIO* runtime_io = &scoped_io->runtime_io_;
  uint32_t nn_id = runtime_io->get_nn_id();
  SemResQueue sem_res_queue;
  {
    tensorflow::mutex_lock lock(mutex_eg_);
    TF_RETURN_IF_ERROR(start_model_unsafe(nn_id));
    uint32_t active_nn_id = NRT_INVALID_NN_ID;
    std::shared_ptr<xla::Semaphore> sem;
    TF_RETURN_IF_ERROR(get_active(&active_nn_id, &sem, nn_id));
    runtime_io->set_nn_id(active_nn_id);
    sem_res_queue.push(sem->ScopedAcquire(1));
    TF_RETURN_IF_ERROR(runtime_.infer_post(runtime_io));
  }
  return runtime_.infer_wait(runtime_io);
}

Status NeuronDevice::infer_with_profiling(ScopedRuntimeIO* scoped_io,
                                          ProfilerInterface* profile) {
  RuntimeIO* runtime_io = &scoped_io->runtime_io_;
  uint32_t nn_id = runtime_io->get_nn_id();
  tensorflow::mutex_lock lock(mutex_eg_);
  TF_RETURN_IF_ERROR(start_model_unsafe(nn_id));
  if (profile->enabled_) profile->start_session(nrtd_address_, nn_id);
  Status status_post = runtime_.infer_post(runtime_io);
  Status status_wait = runtime_.infer_wait(runtime_io);
  if (profile->enabled_) profile->stop_session();
  TF_RETURN_IF_ERROR(status_post);
  return status_wait;
}

void NeuronDevice::clear(bool from_global_state) {
  tensorflow::mutex_lock lock(mutex_eg_);
  if (closed_) {
    return;
  }
  if (from_global_state) {
    closed_ = true;
  }
  for (const auto& nn_id_pair : nn_id_to_all_nn_ids_) {
    const uint32_t nn_id = nn_id_pair.first;
    const std::vector<uint32_t>& all_nn_ids = nn_id_pair.second;
    if (running(nn_id)) {
      // stop all models
      for (const uint32_t nid : all_nn_ids) {
        TF_LOG_IF_ERROR(runtime_.stop(nid));
      }
    }
    // unload all models
    for (const uint32_t nid : all_nn_ids) {
      TF_LOG_IF_ERROR(runtime_.unload(nid, from_global_state));
    }
    VLOG(1) << "unload from NeuronDevice::clear";
  }
  for (const uint32_t eg_id : vec_eg_id_) {
    TF_LOG_IF_ERROR(runtime_.destroy_eg(eg_id, from_global_state));
  }
  VLOG(1) << "destroy_eg from NeuronDevice::clear";
  if (!from_global_state) {
    set_running(NRT_INVALID_NN_ID);
    nn_id_to_all_nn_ids_.clear();
    vec_eg_id_.clear();
  }
}

Status NeuronDevice::start_model_unsafe(const uint32_t nn_id) {
  if (TF_PREDICT_FALSE(closed_)) {
    return errors::Aborted("neuron_device is closed");
  }
  if (TF_PREDICT_FALSE(!running(nn_id) && is_busy())) {
    // if nn_id is not running, stop the current running model
    std::queue<RuntimeStopper> stopper_queue;
    for (const uint32_t nid : nn_id_to_all_nn_ids_[nn_get_current_running()]) {
      stopper_queue.emplace();
      TF_RETURN_IF_ERROR(runtime_.post_stop(&stopper_queue.back(), nid));
    }
    for (const uint32_t nid : nn_id_to_all_nn_ids_[nn_get_current_running()]) {
      TF_RETURN_IF_ERROR(runtime_.wait_stop(&stopper_queue.front()));
      stopper_queue.pop();
      VLOG(1) << "stopped model " << nid;
    }
    set_running(NRT_INVALID_NN_ID);
  }
  if (TF_PREDICT_FALSE(!is_busy())) {
    // if no model is running, start nn_id
    std::queue<RuntimeStarter> starter_queue;
    for (const uint32_t nid : nn_id_to_all_nn_ids_[nn_id]) {
      starter_queue.emplace();
      TF_RETURN_IF_ERROR(runtime_.post_start(&starter_queue.back(), nid));
    }
    for (const uint32_t nid : nn_id_to_all_nn_ids_[nn_id]) {
      TF_RETURN_IF_ERROR(runtime_.wait_start(&starter_queue.front()));
      starter_queue.pop();
      VLOG(1) << "started model " << nid;
    }
    set_running(nn_id);
  }
  return Status::OK();
}

inline bool NeuronDevice::is_busy() {
  return running_nn_id_ != NRT_INVALID_NN_ID;
}

inline bool NeuronDevice::running(uint32_t nn_id) {
  return running_nn_id_ == nn_id && NRT_INVALID_NN_ID != running_nn_id_;
}

inline uint32_t NeuronDevice::nn_get_current_running() {
  return running_nn_id_;
}

inline void NeuronDevice::set_running(uint32_t nn_id) {
  running_nn_id_ = nn_id;
}

Status NeuronDevice::get_active(uint32_t* active_nn_id,
                                std::shared_ptr<xla::Semaphore>* sem,
                                const uint32_t nn_id) {
  if (TF_PREDICT_FALSE(!nn_id_to_all_nn_ids_.count(nn_id))) {
    return errors::InvalidArgument("no active id can be found from nn id ",
                                   nn_id);
  }
  size_t idx = nn_id_to_active_idx_[nn_id];
  nn_id_to_active_idx_[nn_id] = (idx + 1) % nn_id_to_all_nn_ids_[nn_id].size();
  *active_nn_id = nn_id_to_all_nn_ids_[nn_id][idx];
  *sem = nn_id_to_sems_[nn_id][idx];
  return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow

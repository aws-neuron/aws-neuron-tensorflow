/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#include <sys/mman.h>
#include <fcntl.h>
#ifdef NEURONTFSERV
#include <csignal>
#endif  // NEURONTFSERV
#include "device.h"


namespace tensorflow {
namespace neuron {


#define TF_LOG_IF_ERROR(status) {   \
    if (!(status).ok()) {           \
        LOG(ERROR) << (status);     \
    }                               \
}


NeuronDeviceManager global_neuron_device_manager;


#ifdef NEURONTFSERV
void sigint_handler(int sig) {
    global_neuron_device_manager.clear();
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
    std::raise(sig);
}
#endif // NEURONTFSERV


static std::string gen_shm_name(uint32_t nn_id) {
    std::string filename = "/neuron_clib_";
    filename += std::to_string(nn_id);
    for (size_t i = 0; i < 64; ++i) {
        if (Env::Default()->CreateUniqueFileName(&filename, "")) {
            return filename;
        }
        Env::Default()->SleepForMicroseconds(1);
    }
    return "";
}

Status SharedMemoryManager::initialize(const std::string &nrtd_address,
                                       const uint32_t nn_id,
                                       const std::vector<size_t> &input_tensor_sizes,
                                       const std::vector<size_t> &output_tensor_sizes) {
    TF_RETURN_IF_ERROR(runtime_.initialize(nrtd_address));
    TF_RETURN_IF_ERROR(init_vectors(&shm_.input_paths_, &shm_.input_ptrs_, &shm_.input_sizes_,
                                    &nrt_input_paths_, input_tensor_sizes, nn_id));
    TF_RETURN_IF_ERROR(init_vectors(&shm_.output_paths_, &shm_.output_ptrs_, &shm_.output_sizes_,
                                    &nrt_output_paths_, output_tensor_sizes, nn_id));
    for (size_t idx = 0; idx < shm_.input_paths_.size(); ++idx) {
        VLOG(1) << "input shared memory " << shm_.input_paths_[idx]
                << " ready at address " << shm_.input_ptrs_[idx];
    }
    for (size_t idx = 0; idx < shm_.output_paths_.size(); ++idx) {
        VLOG(1) << "output shared memory " << shm_.output_paths_[idx]
                << " ready at address " << shm_.output_ptrs_[idx];
    }
    shm_.enabled_ = true;
    return Status::OK();
}

Status SharedMemoryManager::init_vectors(std::vector<std::string> *names,
                                         std::vector<void*> *ptrs,
                                         std::vector<size_t> *sizes,
                                         std::vector<std::string> *nrt_paths,
                                         const std::vector<size_t> &tensor_sizes,
                                         const uint32_t nn_id) {
    for (size_t size : tensor_sizes) {
        std::string name = gen_shm_name(nn_id);
        if (name.empty()) {
            return errors::Internal("cannot generate unique file name for shared memory");
        }
        int shm_fd = ::shm_open(name.c_str(), O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
        SYS_FAIL_RETURN(shm_fd < 0, "shm_open");
        names->push_back(name);
        SYS_FAIL_RETURN(::ftruncate(shm_fd, size) < 0, "ftruncate");
        void *ptr = ::mmap(0, size, PROT_WRITE, MAP_SHARED, shm_fd, 0);
        SYS_FAIL_RETURN(nullptr == ptr, "mmap");
        ptrs->push_back(ptr);
        sizes->push_back(size);
        TF_RETURN_IF_ERROR(runtime_.shm_map(name, PROT_READ | PROT_WRITE));
        nrt_paths->push_back(name);
    }
    return Status::OK();
}

SharedMemoryManager::~SharedMemoryManager() {
    for (const auto &path : nrt_input_paths_) {
        TF_LOG_IF_ERROR(runtime_.shm_unmap(path, PROT_READ | PROT_WRITE));
    }
    nrt_input_paths_.clear();
    for (size_t idx = 0; idx < shm_.input_ptrs_.size(); ++idx) {
        SYS_FAIL_LOG(munmap(shm_.input_ptrs_[idx], shm_.input_sizes_[idx]) < 0, "munmap");
    }
    shm_.input_ptrs_.clear();
    for (const auto &name : shm_.input_paths_) {
        SYS_FAIL_LOG(shm_unlink(name.c_str()) < 0, "shm_unlink");
    }
    shm_.input_paths_.clear();
    for (const auto &path : nrt_output_paths_) {
        TF_LOG_IF_ERROR(runtime_.shm_unmap(path, PROT_READ | PROT_WRITE));
    }
    nrt_output_paths_.clear();
    for (size_t idx = 0; idx < shm_.output_ptrs_.size(); ++idx) {
        SYS_FAIL_LOG(munmap(shm_.output_ptrs_[idx], shm_.output_sizes_[idx]) < 0, "munmap");
    }
    shm_.output_ptrs_.clear();
    for (const auto &name : shm_.output_paths_) {
        SYS_FAIL_LOG(shm_unlink(name.c_str()) < 0, "shm_unlink");
    }
    shm_.output_paths_.clear();
}


static std::string remove_pattern(std::string data, const std::string &pattern) {
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

NeuronDeviceManager::~NeuronDeviceManager() {
    tensorflow::mutex_lock lock(global_mutex_);
    clear();
}

Status NeuronDeviceManager::initialize(int64_t opt_device_size) {
    if (!path_set_) {
        // append /opt/aws/neuron/bin to PATH
        std::string env_path = env_get("PATH", "");
        setenv("PATH", (env_path + ":/opt/aws/neuron/bin").c_str(), 1);
        path_set_ = true;
    }

    // neuron-rtd address
    nrtd_address_ = env_get("NEURON_RTD_ADDRESS", "unix:/run/neuron.sock");

    // get number of neuron cores from comma-separated list of integers
    std::string neuron_device_sizes_raw = env_get("NEURONCORE_GROUP_SIZES", "");
    if (neuron_device_sizes_raw.empty()) {
        TF_RETURN_IF_ERROR(init_default_device(opt_device_size));
    } else {
        // remove [ and ]
        std::string neuron_device_sizes = remove_pattern(neuron_device_sizes_raw, "[");
        neuron_device_sizes = remove_pattern(neuron_device_sizes, "]");

        std::vector<int> num_cores_req_vector;
        std::stringstream neuron_device_sizes_stream(neuron_device_sizes);
        for (size_t idx = 0; idx < MAX_NUM_CORES; ++idx) {
            if (!neuron_device_sizes_stream.good()) {
                break;
            }
            std::string substr;
            std::getline(neuron_device_sizes_stream, substr, ',');
            if (substr.empty()) {
                continue;
            }
            int num_cores_req = stoi_no_throw(substr);
            if (num_cores_req < 0 || num_cores_req > 64) {
                LOG(WARNING) << "NEURONCORE_GROUP_SIZES=" << neuron_device_sizes_raw
                             << " looks ill-formatted. Falling back to initializing"
                             << " a default NeuronCore Group.";
                num_cores_req_vector.clear();
                break;
            }
            num_cores_req_vector.push_back(num_cores_req);
        }
        if (num_cores_req_vector.empty()) {
            TF_RETURN_IF_ERROR(init_default_device(opt_device_size));
        } else {
            TF_RETURN_IF_ERROR(init_devices(num_cores_req_vector));
        }
    }
    ready_ = true;
    return Status::OK();
}

Status NeuronDeviceManager::init_devices(const std::vector<int> &num_cores_req_vector) {
    Status status = errors::Internal("No NeuronCore Group can be initialized.");
    for (size_t idx = 0; idx < num_cores_req_vector.size(); ++idx) {
        int num_cores_req = num_cores_req_vector[idx];
        status = device_array_[idx].initialize(nrtd_address_, num_cores_req);
        if (!status.ok()) {
            LOG(WARNING) << "Cannot initialize NeuronCore Group with " << num_cores_req
                         << " cores; stopping initialization.";
            break;
        }
        ++num_devices_;
        VLOG(1) << "successfully initialized NeuronCore Group of size " << num_cores_req;
    }
    if (0 == num_devices_) {
        return status;
    }
    return Status::OK();
}

Status NeuronDeviceManager::init_default_device(int64_t opt_device_size) {
    if (opt_device_size < 0 || opt_device_size > 64) {
        // device size looks wrong -- just get the largest ncg possible
        Status status = device_array_[0].initialize(nrtd_address_, DEFAULT_NUM_CORES);
        num_devices_ = status.ok() ? 1 : 0;
        return status;
    } else {
        // get one full Inferentia by default
        if (opt_device_size == 1) {
            std::vector<int> num_cores_req_vector({1, 1, 1, 1});
            TF_RETURN_IF_ERROR(init_devices(num_cores_req_vector));
        } else if (opt_device_size == 2) {
            std::vector<int> num_cores_req_vector({2, 2});
            TF_RETURN_IF_ERROR(init_devices(num_cores_req_vector));
        } else {
            // search for the largest possible ncg ... sorry
            Status status = errors::Internal("No NeuronCore Group can be initialized.");
            for (int num_cores = opt_device_size; num_cores >= MIN_NUM_CORES; --num_cores) {
                status = device_array_[0].initialize(nrtd_address_, num_cores);
                if (status.ok()) {
                    num_devices_ = 1;
                    return status;
                }
            }
            num_devices_ = 0;
            return status;
        }
    }
    return Status::OK();
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
    num_devices_ = 0;
    device_index_ = 0;
    ready_ = false;
    VLOG(1) << "NeuronDeviceManager is cleared";
}

Status NeuronDeviceManager::apply_for_device(NeuronDevice **device,
                                             int64_t opt_device_size) {
    tensorflow::mutex_lock lock(global_mutex_);
    if (!ready_) {
        TF_RETURN_IF_ERROR(initialize(opt_device_size));
#ifdef NEURONTFSERV
        std::signal(SIGINT, sigint_handler);
        std::signal(SIGTERM, sigint_handler);
#endif // NEURONTFSERV
    }

    *device = &device_array_[device_index_];
    ++device_index_;
    if (device_index_ >= num_devices_) {
        device_index_ = 0;
    }
    return Status::OK();
}

Status NeuronDevice::initialize(const std::string &nrtd_address, const int num_cores_req) {
    nrtd_address_ = nrtd_address;
    TF_RETURN_IF_ERROR(runtime_.initialize(nrtd_address_));
    TF_RETURN_IF_ERROR(runtime_.create_eg(&eg_id_, &num_cores_, num_cores_req));
    create_eg_done_ = true;
    running_nn_id_ = NRT_INVALID_NN_ID;
    return Status::OK();
}

Status NeuronDevice::load(uint32_t *nn_id, const StringPiece &executable,
                          const uint32_t timeout, const uint32_t ninfer) {
    TF_RETURN_IF_ERROR(runtime_.load(nn_id, eg_id_, executable, timeout, ninfer));
    tensorflow::mutex_lock lock(mutex_eg_);
    nn_id_set_.insert(*nn_id);
    return Status::OK();
}

void NeuronDevice::unload(const uint32_t nn_id) {
    {
        tensorflow::mutex_lock lock(mutex_eg_);
        if (!nn_id_set_.count(nn_id)) {
            VLOG(1) << "model " << nn_id << " is not loaded";
            return;
        }
        nn_id_set_.erase(nn_id);
        // stop
        if (running(nn_id)) {
            TF_LOG_IF_ERROR(runtime_.stop(nn_id));
            set_running(NRT_INVALID_NN_ID);
        }
    }

    // unload
    if (NRT_INVALID_NN_ID != nn_id) {
        TF_LOG_IF_ERROR(runtime_.unload(nn_id));
    }
    VLOG(1) << "unload: number of NEFFs: " << num_executable();
}

Status NeuronDevice::infer(std::vector<Tensor*> *output_tensors, Timestamps *timestamps,
                           ProfilerInterface *profile, const uint32_t nn_id,
                           AttrList &input_names, AttrList &output_names,
                           const std::vector<const Tensor*> &input_tensors,
                           const SharedMemory &shm) {
    tensorflow::mutex_lock lock(mutex_eg_);
    TF_RETURN_IF_ERROR(start_model(nn_id));
    if (profile->enabled_) profile->start_session(nrtd_address_, nn_id);
    Status status = runtime_.infer(output_tensors, timestamps, nn_id,
                                   input_names, output_names, input_tensors, shm);
    if (profile->enabled_) profile->stop_session();
    return status;
}

Status NeuronDevice::infer_post(NMGROutputs *nmgr_outputs, SemResQueue *sem_res_queue,
                                xla::Semaphore *infer_sem, Timestamps *timestamps,
                                const uint32_t nn_id, AttrList &input_names,
                                const std::vector<const Tensor*> &input_tensors) {
    tensorflow::mutex_lock lock(mutex_eg_);
    sem_res_queue->push(infer_sem->ScopedAcquire(1));
    return infer_post_unsafe(nmgr_outputs, timestamps, nn_id, input_names, input_tensors);
}

void NeuronDevice::acquire_mutex(std::queue<tensorflow::mutex_lock> *mutex_lock_queue) {
    mutex_lock_queue->emplace(mutex_eg_);
}

Status NeuronDevice::infer_post_unsafe(NMGROutputs *nmgr_outputs, Timestamps *timestamps,
                                       const uint32_t nn_id, AttrList &input_names,
                                       const std::vector<const Tensor*> &input_tensors) {
    TF_RETURN_IF_ERROR(start_model(nn_id));
    return runtime_.infer_post(nmgr_outputs, timestamps, nn_id, input_names, input_tensors);
}

Status NeuronDevice::infer_wait(std::vector<Tensor*> *output_tensors, Timestamps *timestamps,
                                const NMGROutputs &nmgr_outputs, AttrList &output_names) {
    return runtime_.infer_wait(output_tensors, timestamps, nmgr_outputs, output_names);
}

void NeuronDevice::clear() {
    tensorflow::mutex_lock lock(mutex_eg_);
    for (uint32_t nn_id : nn_id_set_) {
        if (running(nn_id)) {
            TF_LOG_IF_ERROR(runtime_.stop(nn_id));
            set_running(NRT_INVALID_NN_ID);
        }
        TF_LOG_IF_ERROR(runtime_.unload(nn_id));
        VLOG(1) << "unload from NeuronDevice::clear";
    }
    nn_id_set_.clear();
    if (create_eg_done_) {
        TF_LOG_IF_ERROR(runtime_.destroy_eg(eg_id_));
        create_eg_done_ = false;
        VLOG(1) << "destroy_eg from NeuronDevice::clear";
    }
}

Status NeuronDevice::start_model(const uint32_t nn_id) {
    if (!create_eg_done_) {
        return errors::Internal("neuron_device is not initialized");
    }
    if (!running(nn_id) && is_busy()) {
        // if nn_id is not running, stop the current running model
        TF_RETURN_IF_ERROR(runtime_.stop(nn_get_current_running()));
        set_running(NRT_INVALID_NN_ID);
    }
    if (!is_busy()) {
        // if no model is running, start nn_id
        TF_RETURN_IF_ERROR(runtime_.start(nn_id));
        set_running(nn_id);
    }
    return Status::OK();
}

bool NeuronDevice::is_busy() {
    return running_nn_id_ != NRT_INVALID_NN_ID;
}

bool NeuronDevice::running(uint32_t nn_id) {
    return running_nn_id_ == nn_id && NRT_INVALID_NN_ID != running_nn_id_;
}

uint32_t NeuronDevice::nn_get_current_running() {
    return running_nn_id_;
}

void NeuronDevice::set_running(uint32_t nn_id) {
    running_nn_id_ = nn_id;
}


std::string env_get(const char *env_var, const char *default_env_var) {
    char *str = std::getenv(env_var);
    return str ? str : default_env_var;
}

int stoi_no_throw(const std::string &str) {
    try {
        return std::stoi(str);
    } catch (std::invalid_argument e) {
        return -1;
    } catch (std::out_of_range e) {
        return -1;
    }
}


}  // namespace neuron
}  // namespace tensorflow
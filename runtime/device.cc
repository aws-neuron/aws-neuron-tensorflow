/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#include <sys/mman.h>
#include <sys/stat.h>
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
    global_neuron_device_manager.clear_from_global_state();
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
    std::raise(sig);
}
#endif // NEURONTFSERV


class ShmFile {
public:
    ShmFile(const std::string &name) {
        shm_open_fd_ = ::shm_open(name.c_str(), O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
        if (shm_open_fd_ >= 0) {
            if (::fchmod(shm_open_fd_, S_IRWXU | S_IRWXG | S_IRWXO) < 0) {
                shm_open_fd_ = -1;
            } else {
                shm_fd_ = shm_open_fd_;
            }
        }
    }
    ~ShmFile() {
        if (shm_open_fd_ >= 0) {
            ::close(shm_open_fd_);
        }
    }
    ShmFile(const ShmFile &shm_file) = delete;
    ShmFile &operator=(const ShmFile &shm_file) = delete;
    ShmFile(ShmFile &&) = delete;
    ShmFile &operator=(ShmFile &&) = delete;
    int shm_fd_ = -1;
private:
    int shm_open_fd_ = -1;
};


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
                                       const uint32_t nn_id, const uint32_t max_num_infers,
                                       const std::vector<size_t> &input_tensor_sizes,
                                       const std::vector<size_t> &output_tensor_sizes) {
    tensorflow::mutex_lock lock(mutex_);
    if (enabled_) {
        return Status::OK();
    }
    TF_RETURN_IF_ERROR(runtime_.initialize(nrtd_address));
    uint32_t num_buffers = max_num_infers + 1;  // +1 because batching mode uses one more buffer
    shm_vec_.resize(num_buffers);
    shm_busy_vec_.clear();
    shm_busy_vec_.resize(num_buffers, 0);
    num_shms_ = 0;
    for (auto &shm : shm_vec_) {
        TF_RETURN_IF_ERROR(init_vectors(
            &shm.input_paths_, &shm.input_ptrs_, &shm.input_sizes_,
            &shm.nrt_input_paths_, input_tensor_sizes, nn_id));
        TF_RETURN_IF_ERROR(init_vectors(
            &shm.output_paths_, &shm.output_ptrs_, &shm.output_sizes_,
            &shm.nrt_output_paths_, output_tensor_sizes, nn_id));
        for (size_t idx = 0; idx < shm.input_paths_.size(); ++idx) {
            VLOG(1) << "input shared memory " << shm.input_paths_[idx]
                    << " ready at address " << shm.input_ptrs_[idx];
        }
        for (size_t idx = 0; idx < shm.output_paths_.size(); ++idx) {
            VLOG(1) << "output shared memory " << shm.output_paths_[idx]
                    << " ready at address " << shm.output_ptrs_[idx];
        }
        ++num_shms_;
    }
    enabled_ = true;
    return Status::OK();
}

Status SharedMemoryManager::init_vectors(std::vector<std::string> *names,
                                         std::vector<char*> *ptrs,
                                         std::vector<size_t> *sizes,
                                         std::vector<std::string> *nrt_paths,
                                         const std::vector<size_t> &tensor_sizes,
                                         const uint32_t nn_id) {
    for (size_t size : tensor_sizes) {
        std::string name = gen_shm_name(nn_id);
        if (name.empty()) {
            return errors::Internal("cannot generate unique file name for shared memory");
        }
        ShmFile shm_file(name);
        SYS_FAIL_RETURN(shm_file.shm_fd_ < 0, "shm_open");
        names->push_back(name);
        SYS_FAIL_RETURN(::ftruncate(shm_file.shm_fd_, size) < 0, "ftruncate");
        char *ptr = static_cast<char*>(::mmap(0, size, PROT_WRITE, MAP_SHARED, shm_file.shm_fd_, 0));
        SYS_FAIL_RETURN(nullptr == ptr, "mmap");
        ptrs->push_back(ptr);
        sizes->push_back(size);
        TF_RETURN_IF_ERROR(runtime_.shm_map(name, PROT_READ | PROT_WRITE));
        nrt_paths->push_back(name);
    }
    return Status::OK();
}

SharedMemory *SharedMemoryManager::apply_for_shm() {
    if (!enabled_) {
        return nullptr;
    }
    tensorflow::mutex_lock lock(mutex_);
    for (size_t idx = 0; idx < num_shms_; ++idx) {
        if (!shm_busy_vec_[idx]) {
            shm_busy_vec_[idx] = 1;
            return &shm_vec_[idx];
        }
    }
    return nullptr;
}

void SharedMemoryManager::free_shm(SharedMemory *shm) {
    if (!enabled_ || nullptr == shm) {
        return;
    }
    tensorflow::mutex_lock lock(mutex_);
    for (size_t idx = 0; idx < num_shms_; ++idx) {
        if (shm == &shm_vec_[idx]) {
            shm_busy_vec_[idx] = 0;
            return;
        }
    }
}

void SharedMemoryManager::clear() {
    for (auto &shm : shm_vec_) {
        for (const auto &path : shm.nrt_input_paths_) {
            TF_LOG_IF_ERROR(runtime_.shm_unmap(path, PROT_READ | PROT_WRITE));
        }
        shm.nrt_input_paths_.clear();
        for (size_t idx = 0; idx < shm.input_ptrs_.size(); ++idx) {
            SYS_FAIL_LOG(munmap(shm.input_ptrs_[idx], shm.input_sizes_[idx]) < 0, "munmap");
        }
        shm.input_ptrs_.clear();
        for (const auto &name : shm.input_paths_) {
            SYS_FAIL_LOG(shm_unlink(name.c_str()) < 0, "shm_unlink");
        }
        shm.input_paths_.clear();
        for (const auto &path : shm.nrt_output_paths_) {
            TF_LOG_IF_ERROR(runtime_.shm_unmap(path, PROT_READ | PROT_WRITE));
        }
        shm.nrt_output_paths_.clear();
        for (size_t idx = 0; idx < shm.output_ptrs_.size(); ++idx) {
            SYS_FAIL_LOG(munmap(shm.output_ptrs_[idx], shm.output_sizes_[idx]) < 0, "munmap");
        }
        shm.output_ptrs_.clear();
        for (const auto &name : shm.output_paths_) {
            SYS_FAIL_LOG(shm_unlink(name.c_str()) < 0, "shm_unlink");
        }
        shm.output_paths_.clear();
    }
    num_shms_ = 0;
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
    clear_from_global_state();
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

void NeuronDeviceManager::clear_from_global_state() {
    for (size_t idx = 0; idx < num_devices_; ++idx) {
        device_array_[idx].clear(true);
    }
    num_devices_ = 0;
    device_index_ = 0;
    ready_ = false;
    VLOG(1) << "NeuronDeviceManager is cleared from global state";
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
        if (nn_id_to_shm_mgr_.count(nn_id)) {
            nn_id_to_shm_mgr_[nn_id].clear();
        }
        nn_id_to_shm_mgr_.erase(nn_id);
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

Status NeuronDevice::setup_async_io(RuntimeIO *runtime_io, int64_t post_tag) {
    return runtime_.setup_async_io(runtime_io, post_tag);
}

Status NeuronDevice::post_infer_post(RuntimeIO *runtime_io) {
    return runtime_.post_infer_post(runtime_io);
}

Status NeuronDevice::wait_infer_post(RuntimeIO *runtime_io) {
    return runtime_.wait_infer_post(runtime_io);
}

Status NeuronDevice::infer(RuntimeIO *runtime_io, Timestamps *timestamps,
                           ProfilerInterface *profile, const uint32_t nn_id) {
    tensorflow::mutex_lock lock(mutex_eg_);
    TF_RETURN_IF_ERROR(start_model_unsafe(nn_id));
    if (profile->enabled_) profile->start_session(nrtd_address_, nn_id);
    Status status_post = runtime_.infer_post(runtime_io);
    Status status_wait = runtime_.infer_wait(runtime_io);
    if (profile->enabled_) profile->stop_session();
    TF_RETURN_IF_ERROR(status_post);
    TF_RETURN_IF_ERROR(status_wait);
    return runtime_io->finish();
}

Status NeuronDevice::infer_post(RuntimeIO *runtime_io, SemResQueue *sem_res_queue,
                                xla::Semaphore *infer_sem, Timestamps *timestamps,
                                const uint32_t nn_id) {
    tensorflow::mutex_lock lock(mutex_eg_);
    sem_res_queue->push(infer_sem->ScopedAcquire(1));
    return infer_post_unsafe(runtime_io, timestamps, nn_id);
}

void NeuronDevice::acquire_mutex(std::queue<tensorflow::mutex_lock> *mutex_lock_queue) {
    mutex_lock_queue->emplace(mutex_eg_);
}

Status NeuronDevice::infer_post_unsafe(RuntimeIO *runtime_io, Timestamps *timestamps,
                                       const uint32_t nn_id) {
    TF_RETURN_IF_ERROR(start_model_unsafe(nn_id));
    if (nullptr != timestamps) timestamps->mark_above_nrtd_infer();
    return runtime_.infer_post(runtime_io);
}

Status NeuronDevice::infer_wait(RuntimeIO *runtime_io, Timestamps *timestamps) {
    TF_RETURN_IF_ERROR(runtime_.infer_wait(runtime_io));
    if (nullptr != timestamps) timestamps->mark_below_nrtd_infer();
    return Status::OK();
}

Status NeuronDevice::init_shm_mgr(SharedMemoryManager **shm_mgr,
                                  const uint32_t nn_id, const uint32_t max_num_infers,
                                  const std::vector<size_t> input_tensor_sizes,
                                  const std::vector<size_t> output_tensor_sizes) {
    std::string unix_prefix = "unix:";
    if (nrtd_address_.compare(0, unix_prefix.size(), unix_prefix)) {
        return errors::InvalidArgument("shared memory requires using unix socket");
    }
    tensorflow::mutex_lock lock(mutex_eg_);
    nn_id_to_shm_mgr_.emplace(std::piecewise_construct, std::forward_as_tuple(nn_id), std::forward_as_tuple());
    TF_RETURN_IF_ERROR(nn_id_to_shm_mgr_[nn_id].initialize(
        nrtd_address_, nn_id, max_num_infers, input_tensor_sizes, output_tensor_sizes));
    *shm_mgr = &nn_id_to_shm_mgr_[nn_id];
    return Status::OK();
}

void NeuronDevice::clear(bool from_global_state) {
    tensorflow::mutex_lock lock(mutex_eg_);
    for (uint32_t nn_id : nn_id_set_) {
        if (running(nn_id)) {
            TF_LOG_IF_ERROR(runtime_.stop(nn_id));
            set_running(NRT_INVALID_NN_ID);
        }
        TF_LOG_IF_ERROR(runtime_.unload(nn_id, from_global_state));
        VLOG(1) << "unload from NeuronDevice::clear";
    }
    for (auto &item : nn_id_to_shm_mgr_) {
        item.second.clear();
    }
    nn_id_to_shm_mgr_.clear();
    nn_id_set_.clear();
    if (create_eg_done_) {
        TF_LOG_IF_ERROR(runtime_.destroy_eg(eg_id_, from_global_state));
        create_eg_done_ = false;
        VLOG(1) << "destroy_eg from NeuronDevice::clear";
    }
}

Status NeuronDevice::start_ping(const uint32_t nn_id) {
    return runtime_.start_ping(nn_id);
}

Status NeuronDevice::start_model_unsafe(const uint32_t nn_id) {
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

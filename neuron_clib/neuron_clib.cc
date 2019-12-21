/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#include <sstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <grpcpp/grpcpp.h>
#include "tensorflow/core/platform/env.h"
#include "neuron_clib.h"
#include "nmgr.pb.h"
#ifdef NEURONTFSERV
#include <csignal>
#endif  // NEURONTFSERV


namespace tensorflow {
namespace neuron {


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


Status SharedMemory::initialize(const std::string &nrtd_address, uint32_t nn_id) {
    name_ = gen_shm_name(nn_id);
    if (name_.empty()) {
        return errors::Internal("cannot generate unique file name for shared memory");
    }
    TF_RETURN_IF_ERROR(init_stub(&stub_, nrtd_address));
    int shm_fd = ::shm_open(name_.c_str(), O_CREAT | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
    SYS_FAIL_RETURN(shm_fd < 0, "shm_open");
    shm_open_done_ = true;
    SYS_FAIL_RETURN(::ftruncate(shm_fd, size_) < 0, "ftruncate");
    ptr_ = ::mmap(0, size_, PROT_WRITE, MAP_SHARED, shm_fd, 0);
    SYS_FAIL_RETURN(nullptr == ptr_, "mmap");
    nrt::shm_map_request request;
    request.set_path(name_);
    request.set_mmap_prot(PROT_READ | PROT_WRITE);
    nrt::shm_map_response response;
    grpc::Status status = NRT_GRPC(stub_->shm_map, request, &response);
    NRT_CHECK_RETURN("shm_map", status, response);
    shm_map_done_ = true;
    return Status::OK();
}

void SharedMemory::clear() {
    if (shm_map_done_) {
        nrt::shm_unmap_request request;
        request.set_path(name_);
        request.set_mmap_prot(PROT_READ | PROT_WRITE);
        nrt::shm_unmap_response response;
        grpc::Status status = NRT_GRPC(stub_->shm_unmap, request, &response);
        NRT_CHECK_LOG("shm_unmap", status, response);
        if (status.ok() && 0 == response.status().code()) {
            shm_map_done_ = false;
        }
    }
    if (nullptr != ptr_) {
        int ret = munmap(ptr_, size_);
        SYS_FAIL_LOG(ret < 0, "munmap");
        if (ret >= 0) {
            ptr_ = nullptr;
        }
    }
    if (shm_open_done_) {
        int ret = shm_unlink(name_.c_str());
        SYS_FAIL_LOG(ret < 0, "shm_unlink");
        if (ret >= 0) {
            shm_open_done_ = false;
        }
    }
}


static std::string remove_pattern(std::string data, const std::string &pattern) {
    size_t string_length = data.size();
    for (size_t idx = 0; idx < string_length; ++idx) {
        size_t pos = data.find(pattern, pos);
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

    // stub
    nrtd_address_ = env_get("NEURON_RTD_ADDRESS", "unix:/run/neuron.sock");
    TF_RETURN_IF_ERROR(init_stub(&stub_, nrtd_address_));

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

Status NeuronDeviceManager::clear_if_empty() {
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
    return Status::OK();
}

void NeuronDeviceManager::clear() {
    for (size_t idx = 0; idx < num_devices_; ++idx) {
        tensorflow::mutex_lock lock(device_array_[idx].mutex_eg_);
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

Status NeuronDevice::initialize(const std::string &nrtd_address, int num_cores_req) {
    TF_RETURN_IF_ERROR(init_stub(&stub_, nrtd_address));
    nrt::create_eg_request request;
    if (num_cores_req >= 0) {
        request.set_nc_count((uint32_t)num_cores_req);
    }
    nrt::create_eg_response response;
    grpc::Status status = NRT_GRPC(stub_->create_eg, request, &response);
    if (!status.ok() && grpc::StatusCode::UNAVAILABLE == status.error_code()) {
        std::string message(" is unavailable. Is neuron-rtd running?");
        std::string unix_prefix("unix:");
        size_t start = nrtd_address.find(unix_prefix);
        if (0 == start) {
            message += " Is socket ";
            message += nrtd_address.substr(start + unix_prefix.length());
            message += " writable?";
        }
        return errors::Unavailable("grpc server ", nrtd_address, message);
    }
    NRT_CHECK_RETURN("create_eg", status, response);
    num_cores_ = response.nc_count();
    eg_id_ = response.h_eg().id();
    create_eg_done_ = true;
    running_nn_id_ = NRT_INVALID_NN_ID;
    return Status::OK();
}

Status NeuronDevice::is_valid() {
    return create_eg_done_ ? Status::OK()
                           : errors::Internal("neuron_device is not initialized");
}

void NeuronDevice::clear() {
    for (uint32_t nn_id : nn_id_set_) {
        // stop
        if (running(nn_id)) {
            nrt::stop_request request;
            request.mutable_h_nn()->set_id(nn_id);
            nrt::stop_response response;
            grpc::Status status = NRT_GRPC(stub_->stop, request, &response);
            NRT_CHECK_LOG("stop", status, response);
            set_running(NRT_INVALID_NN_ID);
        }

        // unload
        nrt::unload_request request;
        request.mutable_h_nn()->set_id(nn_id);
        nrt::unload_response response;
        grpc::Status status = NRT_GRPC(stub_->unload, request, &response);
        NRT_CHECK_LOG("unload", status, response);
        VLOG(1) << "unload from NeuronDevice::clear";
    }
    nn_id_set_.clear();
    if (create_eg_done_) {
        // destroy_eg
        nrt::destroy_eg_request request;
        request.mutable_h_eg()->set_id(eg_id_);
        nrt::destroy_eg_response response;
        grpc::Status status = NRT_GRPC(stub_->destroy_eg, request, &response);
        NRT_CHECK_LOG("destroy_eg", status, response);
        create_eg_done_ = false;
        VLOG(1) << "destroy_eg from NeuronDevice::clear";
    }
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


static std::string uint64_to_string(uint64 number) {
    std::ostringstream oss;
    oss << number;
    return oss.str();
}

std::string FALTimestamps::timing_string() {
    std::string result("NeuronOp enter timestamp: ");
    result += uint64_to_string(enter_);
    result += time_unit_;
    result += ", preprocessing time ";
    result += uint64_to_string(above_nrtd_infer_ - enter_);
    result += time_unit_;
    result += ", neuron-rtd infer time ";
    result += uint64_to_string(below_nrtd_infer_ - above_nrtd_infer_);
    result += time_unit_;
    result += ", postprocessing time ";
    result += uint64_to_string(exit_ - below_nrtd_infer_);
    result += time_unit_;
    return result;
}

uint64 FALTimestamps::now() {
    return Env::Default()->NowMicros();
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

Status init_stub(std::unique_ptr<nrt::nmgr_v1::Stub> *stub,
                 const std::string &nrtd_address) {
    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    ch_args.SetMaxSendMessageSize(-1);
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        nrtd_address, grpc::InsecureChannelCredentials(), ch_args);
    if (!channel) {
        return errors::Unavailable(
            "cannot establish grpc channel to neuron-rtd server");
    }
    *stub = nrt::nmgr_v1::NewStub(channel);
    if (!(*stub)) {
        return errors::Unavailable("cannot create stub");
    }
    return Status::OK();
}


}  // namespace neuron
}  // namespace tensorflow

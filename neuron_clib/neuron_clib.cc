/* Copyright 2019, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#include <sstream>
#include <sys/mman.h>
#include <fcntl.h>
#include "uuid/uuid.h"
#include <grpcpp/grpcpp.h>
#include "neuron_clib.h"


namespace tensorflow {
namespace kaena {


static std::string gen_shm_name() {
    // typedef unsigned char uuid_t[16];
    uuid_t uuid;

    // generate
    uuid_generate_time(uuid);

    // unparse (to string)
    char uuid_str[128];  // ex. "1b4e28ba-2fa1-11d2-883f-0016d3cca427" + "\0"
    uuid_unparse_lower(uuid, uuid_str);

    return "/neuron_clib_" + std::string(uuid_str);
}


Status SharedMemory::initialize(const std::unique_ptr<nrt::nmgr_v1::Stub> &stub) {
    name_ = gen_shm_name();
    int shm_fd = ::shm_open(name_.c_str(), O_CREAT | O_RDWR, S_IRWXU | S_IRWXG);
    SYS_FAIL_RETURN(shm_fd < 0, "shm_open");
    shm_open_done_ = true;
    SYS_FAIL_RETURN(ftruncate(shm_fd, size_) < 0, "ftruncate");
    ptr_ = ::mmap(0, size_, PROT_WRITE, MAP_SHARED, shm_fd, 0);
    SYS_FAIL_RETURN(nullptr == ptr_, "mmap");
    grpc::Status status;
    grpc::ClientContext context;
    nrt::shm_map_request shm_map_request;
    shm_map_request.set_path(name_);
    shm_map_request.set_mmap_prot(PROT_READ | PROT_WRITE);
    nrt::shm_map_response shm_map_response;
    status = stub->shm_map(&context, shm_map_request, &shm_map_response);
    NRT_CHECK_RETURN("shm_map", status, shm_map_response);
    shm_map_done_ = true;
    return Status::OK();
}

void SharedMemory::clear(const std::unique_ptr<nrt::nmgr_v1::Stub> &stub) {
    if (shm_map_done_) {
        nrt::shm_unmap_request request;
        request.set_path(name_);
        request.set_mmap_prot(PROT_READ | PROT_WRITE);
        grpc::Status status;
        grpc::ClientContext context;
        nrt::shm_unmap_response response;
        status = stub->shm_unmap(&context, request, &response);
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


SharedMemoryAllocator::SharedMemoryAllocator(SharedMemory *shared_memory)
    : shared_memory_(shared_memory) {}
SharedMemoryAllocator::~SharedMemoryAllocator() = default;

std::string SharedMemoryAllocator::Name() { return "neuron_shared_memory"; }

void* SharedMemoryAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
    return shared_memory_->ptr();
}

void SharedMemoryAllocator::DeallocateRaw(void *ptr) {
}


static std::string remove_pattern(std::string data, const std::string &pattern) {
    // Get the first occurrence
	size_t pos = data.find(pattern);

	// Repeat till end is reached
	while( pos != std::string::npos) {
		// Replace this occurrence of Sub String
		data.replace(pos, pattern.size(), "");
		// Get the next occurrence from the current position
		pos = data.find(pattern, pos);
	}
    return data;
}


Status NeuronDeviceManager::initialize() {
    // append /opt/aws/neuron/bin to PATH
    std::string env_path = env_get("PATH", "");
    setenv("PATH", (env_path + ":/opt/aws/neuron/bin").c_str(), 1);

    // stub
    std::string nrtd_address = env_get("NEURON_RTD_ADDRESS",
                                       "unix:/run/neuron.sock");

    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    ch_args.SetMaxSendMessageSize(-1);
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        nrtd_address, grpc::InsecureChannelCredentials(), ch_args);
    if (nullptr == channel) {
        return errors::Unavailable(
            "cannot establish grpc channel to neuron-rtd server");
    }
    stub_ = nrt::nmgr_v1::NewStub(channel);
    if (nullptr == stub_) {
        return errors::Unavailable("cannot create stub");
    }

    // get number of neuron cores from comma-separated list of integers
    std::string neuron_device_sizes_raw = env_get("NEURON_DEVICE_SIZES", "");
    if ("" == neuron_device_sizes_raw) {
        TF_RETURN_IF_ERROR(init_default_device(nrtd_address));
    } else {
        // remove [ and ]
        std::string neuron_device_sizes = remove_pattern(neuron_device_sizes_raw, "[");
        neuron_device_sizes = remove_pattern(neuron_device_sizes, "]");

        std::vector<uint32_t> num_cores_vector;
        std::stringstream neuron_device_sizes_stream(neuron_device_sizes);
        while (neuron_device_sizes_stream.good()) {
            std::string substr;
            std::getline(neuron_device_sizes_stream, substr, ',');
            if (substr.empty()) {
                continue;
            }
            int int_num_cores = stoi_no_throw(substr);
            if (int_num_cores < 0 || int_num_cores > 64) {
                LOG(WARNING) << "NEURON_DEVICE_SIZES=" << neuron_device_sizes_raw
                             << " looks ill-formatted. Falling back to initializing"
                             << " a default Neuron device.";
                num_cores_vector.clear();
                break;
            }
            num_cores_vector.push_back((uint32_t)int_num_cores);
        }
        if (num_cores_vector.empty()) {
            TF_RETURN_IF_ERROR(init_default_device(nrtd_address));
        } else {
            Status status = errors::Internal("No Neuron device can be initialized.");
            for (size_t idx = 0; idx < num_cores_vector.size(); ++idx) {
                uint32_t num_cores = num_cores_vector[idx];
                status = device_array_[idx].initialize(stub_, num_cores, nrtd_address);
                if (!status.ok()) {
                    LOG(WARNING) << "Cannot initialize Neuron device with " << num_cores
                                 << " cores; stopping initialization.";
                    break;
                }
                ++num_devices_;
                VLOG(1) << "successfully initialized Neuron device of size " << num_cores;
            }
            if (0 == num_devices_) {
                return status;
            }
        }
    }
    ready_ = true;
    return Status::OK();
}

Status NeuronDeviceManager::init_default_device(const std::string &nrtd_address) {
    Status status = errors::Internal("No Neuron device can be initialized.");
    for (size_t num_cores = MAX_NUM_CORES; num_cores >= MIN_NUM_CORES; --num_cores) {
        status = device_array_[0].initialize(stub_, num_cores, nrtd_address);
        if (status.ok()) {
            num_devices_ = 1;
            return status;
        }
    }
    num_devices_ = 0;
    return status;
}

bool NeuronDeviceManager::is_empty() {
    bool empty = true;
    for (size_t idx = 0; idx < num_devices_; ++idx) {
        if (0 != device_array_[idx].num_executable()) {
            empty = false;
        }
    }
    return empty;
}

void NeuronDeviceManager::clear() {
    for (size_t idx = 0; idx < num_devices_; ++idx) {
        device_array_[idx].clear(stub_);
    }
    num_devices_ = 0;
    device_index_ = 0;
    ready_ = false;
    VLOG(1) << "NeuronDeviceManager is cleared";
}

NeuronDevice *NeuronDeviceManager::get_device() {
    NeuronDevice *device = &device_array_[device_index_];
    ++device_index_;
    if (device_index_ >= num_devices_) {
        device_index_ = 0;
    }
    return device;
}


Status NeuronDevice::initialize(std::unique_ptr<nrt::nmgr_v1::Stub> &stub,
                                uint32_t num_cores,
                                const std::string &nrtd_address) {
    grpc::Status status;
    grpc::ClientContext context;
    nrt::create_eg_request create_eg_request;
    create_eg_request.set_nc_count(num_cores);
    nrt::create_eg_response create_eg_response;
    status = stub->create_eg(&context, create_eg_request, &create_eg_response);
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
    NRT_CHECK_RETURN("create_eg", status, create_eg_response);
    num_cores_ = num_cores;
    eg_id_ = create_eg_response.h_eg().id();
    create_eg_done_ = true;
    running_nn_id_ = NRT_INVALID_NN_ID;
    return Status::OK();
}

void NeuronDevice::clear(std::unique_ptr<nrt::nmgr_v1::Stub> &stub) {
    grpc::Status status;
    for (uint32_t nn_id : nn_id_set_) {
        // stop
        if (running(nn_id)) {
            grpc::ClientContext context;
            nrt::stop_request stop_request;
            stop_request.mutable_h_nn()->set_id(nn_id);
            nrt::stop_response stop_response;
            status = stub->stop(&context, stop_request, &stop_response);
            NRT_CHECK_LOG("stop", status, stop_response);
            set_running(NRT_INVALID_NN_ID);
        }

        // unload
        grpc::ClientContext context;
        nrt::unload_request unload_request;
        unload_request.mutable_h_nn()->set_id(nn_id);
        nrt::unload_response unload_response;
        status = stub->unload(&context, unload_request, &unload_response);
        NRT_CHECK_LOG("unload", status, unload_response);
    }
    nn_id_set_.clear();
    if (create_eg_done_) {
        // destroy_eg
        grpc::ClientContext context;
        nrt::destroy_eg_request request;
        request.mutable_h_eg()->set_id(eg_id_);
        nrt::destroy_eg_response response;
        status = stub->destroy_eg(&context, request, &response);
        NRT_CHECK_LOG("destroy_eg", status, response);
        create_eg_done_ = false;
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


}  // namespace kaena
}  // namespace tensorflow

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


tensorflow::Status SharedMemory::initialize(
        const std::unique_ptr<nrt::nmgr_v1::Stub> &stub) {
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
    krtd_shm_map_done_ = true;
    return tensorflow::Status::OK();
}

void SharedMemory::clear(const std::unique_ptr<nrt::nmgr_v1::Stub> &stub) {
    if (krtd_shm_map_done_) {
        nrt::shm_unmap_request shm_unmap_request;
        shm_unmap_request.set_path(name_);
        shm_unmap_request.set_mmap_prot(PROT_READ | PROT_WRITE);
        grpc::Status status;
        grpc::ClientContext context;
        nrt::shm_unmap_response shm_unmap_response;
        status = stub->shm_unmap(&context, shm_unmap_request, &shm_unmap_response);
        NRT_CHECK_LOG("shm_unmap", status, shm_unmap_response);
        if (status.ok() && 0 == shm_unmap_response.status().code()) {
            krtd_shm_map_done_ = false;
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


tensorflow::Status NeuronDeviceManager::initialize() {
    // append /opt/aws/neuron/bin to PATH
    std::string env_path = env_get("PATH", "");
    setenv("PATH", (env_path + ":/opt/aws/neuron/bin").c_str(), 1);

    // get krtd stub
    std::string krtd_server = env_get("NEURON_RTD_ADDRESS", "unix:/run/neuron.sock");

    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    ch_args.SetMaxSendMessageSize(-1);
    std::shared_ptr<grpc::Channel> krt_channel = grpc::CreateCustomChannel(
        krtd_server, grpc::InsecureChannelCredentials(), ch_args);
    if (nullptr == krt_channel) {
        return tensorflow::errors::Unavailable(
            "cannot establish grpc channel to neuron-rtd server");
    }
    stub_ = nrt::nmgr_v1::NewStub(krt_channel);
    if (nullptr == stub_) {
        return tensorflow::errors::Unavailable("cannot create stub");
    }

    // get number of neuron cores from comma-separated list of integers
    std::vector<int> num_cores_vector;
    std::string neuron_device_sizes = env_get("NEURON_DEVICE_SIZES", "1");
    std::stringstream neuron_device_sizes_stream(neuron_device_sizes);
    while (neuron_device_sizes_stream.good()) {
        std::string substr;
        std::getline(neuron_device_sizes_stream, substr, ',');
        if (substr.empty()) {
            continue;
        }
        num_cores_vector.push_back(std::stoi(substr));
    }
    if (num_cores_vector.empty()) {
        return tensorflow::errors::InvalidArgument(
            "NEURON_DEVICE_SIZES=", neuron_device_sizes, " is ill-formatted");
    }

    tensorflow::Status status;
    for (size_t idx = 0; idx < num_cores_vector.size(); ++idx) {
        int num_cores = num_cores_vector[idx];
        status = device_array_[idx].initialize(stub_, num_cores, krtd_server);
        if (!status.ok()) {
            return status;
        }
        ++num_devices_;
    }
    ready_ = true;
    return tensorflow::Status::OK();
}

bool NeuronDeviceManager::is_empty() {
    if (num_devices_ > 1) {
        return false;
    }
    bool empty = true;
    for (size_t idx = 0; idx < num_devices_; ++idx) {
        if (0 != device_array_[idx].get_num_executable()) {
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
}

NeuronDevice *NeuronDeviceManager::get_device() {
    NeuronDevice *device = &device_array_[device_index_];
    ++device_index_;
    if (device_index_ >= num_devices_) {
        device_index_ = 0;
    }
    return device;
}


tensorflow::Status NeuronDevice::initialize(
        std::unique_ptr<nrt::nmgr_v1::Stub> &stub, int size,
        const std::string &krtd_server) {
    grpc::Status status;
    grpc::ClientContext context;
    nrt::create_eg_request create_eg_request;
    create_eg_request.set_nc_count(size);
    nrt::create_eg_response create_eg_response;
    status = stub->create_eg(&context, create_eg_request, &create_eg_response);
    if (!status.ok() && grpc::StatusCode::UNAVAILABLE == status.error_code()) {
        std::string message(" is unavailable. Is neuron-rtd running?");
        std::string unix_keyword("unix:");
        size_t socket_start = krtd_server.find(unix_keyword);
        if (0 == socket_start) {
            message += " Is socket ";
            message += krtd_server.substr(socket_start + unix_keyword.length());
            message += " writable?";
        }
        return tensorflow::errors::Unavailable("grpc server ", krtd_server, message);
    }
    NRT_CHECK_RETURN("create_eg", status, create_eg_response);
    krt_eg_id_ = create_eg_response.h_eg().id();
    create_eg_done_ = true;
    krt_nn_id_running_ = NRT_INVALID_NN_ID;
    return tensorflow::Status::OK();
}

void NeuronDevice::clear(std::unique_ptr<nrt::nmgr_v1::Stub> &stub) {
    grpc::Status status;
    for (uint32_t nn_id : krt_h_nn_ids_) {
        // stop
        if (nn_is_running(nn_id)) {
            grpc::ClientContext context;
            nrt::stop_request stop_request;
            stop_request.mutable_h_nn()->set_id(nn_id);
            nrt::stop_response stop_response;
            status = stub->stop(&context, stop_request, &stop_response);
            NRT_CHECK_LOG("stop", status, stop_response);
            nn_set_current_running(NRT_INVALID_NN_ID);
        }

        // unload
        grpc::ClientContext context;
        nrt::unload_request unload_request;
        unload_request.mutable_h_nn()->set_id(nn_id);
        nrt::unload_response unload_response;
        status = stub->unload(&context, unload_request, &unload_response);
        NRT_CHECK_LOG("unload", status, unload_response);
    }
    krt_h_nn_ids_.clear();
    if (create_eg_done_) {
        // destroy_eg
        grpc::ClientContext context;
        nrt::destroy_eg_request destroy_eg_request;
        destroy_eg_request.mutable_h_eg()->set_id(krt_eg_id_);
        nrt::destroy_eg_response destroy_eg_response;
        status = stub->destroy_eg(&context, destroy_eg_request, &destroy_eg_response);
        NRT_CHECK_LOG("destroy_eg", status, destroy_eg_response);
        create_eg_done_ = false;
    }
}

bool NeuronDevice::some_nn_is_running() {
    return krt_nn_id_running_ != 0;
}

bool NeuronDevice::nn_is_running(uint32_t krt_nn_id) {
    return krt_nn_id_running_ == krt_nn_id;
}

uint32_t NeuronDevice::nn_get_current_running() {
    return krt_nn_id_running_;
}

void NeuronDevice::nn_set_current_running(uint32_t krt_nn_id) {
    krt_nn_id_running_ = krt_nn_id;
}


static std::string uint64_to_string(uint64 number) {
    std::ostringstream oss;
    oss << number;
    return oss.str();
}

std::string FALTimestamps::timing_string() {
    std::string result("NeuronOp enter timestamp: ");
    result += uint64_to_string(enter);
    result += time_unit;
    if (above_krtd_infer.size() > 0) {
        result += ", preprocessing time ";
        result += uint64_to_string(above_krtd_infer[0] - enter);
        result += time_unit;
        size_t num_infer = std::min(above_krtd_infer.size(), below_krtd_infer.size());
        for (size_t idx = 0; idx < num_infer; ++idx) {
            if (idx > 1) {
                result += ", inter-processing time ";
                result += uint64_to_string(
                    above_krtd_infer[idx] - below_krtd_infer[idx - 1]);
                result += time_unit;
            }
            result += ", neuron-rtd infer time ";
            result += uint64_to_string(below_krtd_infer[idx] - above_krtd_infer[idx]);
            result += time_unit;
        }
        if (num_infer - 1 < below_krtd_infer.size()) {
            result += ", postprocessing time ";
            result += uint64_to_string(exit - below_krtd_infer[num_infer - 1]);
            result += time_unit;
        }
    }
    return result;
}


std::string env_get(const char *env_var, const char *default_env_var) {
    char *str = std::getenv(env_var);
    return str ? str : default_env_var;
}


}  // namespace kaena
}  // namespace tensorflow

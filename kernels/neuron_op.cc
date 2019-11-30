/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#include <fstream>
#include <sys/types.h>
#include <sys/wait.h>
#include "tensorflow/python/neuron/kernels/neuron_op.h"
#include <grpcpp/grpcpp.h>
#include <queue>
#include "nerr.pb.h"
#ifdef NEURONTFSERV
#include <csignal>
#endif  // NEURONTFSERV


namespace tensorflow {
namespace neuron {


static const size_t EXEC_MAX_CHUNK_SIZE = 1024 * 1024;  // some reasonable number of bytes
static const int64 UNINIT_BATCH_SIZE = -8;  // magic number for uninitialized batch size


NeuronDeviceManager global_neuron_device_manager;


#ifdef NEURONTFSERV
void sigint_handler(int sig) {
    global_neuron_device_manager.clear();
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
    std::raise(sig);
}
#endif // NEURONTFSERV


NeuronOp::NeuronOp(OpKernelConstruction *ctx)
        : OpKernel(ctx), infer_sem_(INFER_SEM_MAX_CAPACITY) {
    VLOG(1) << "calling NeuronOp constructor";
    OP_REQUIRES(ctx, 0 != def().attr().at("executable").s().size(),
                errors::InvalidArgument("neff is invalid"));
    profile_dir_ = env_get("NEURON_PROFILE");
    profile_enabled_ = !profile_dir_.empty();
    if (profile_enabled_) {
        profile_dump_info();
    }
    VLOG(1) << "NeuronOp constructor done";
}


static bool model_config_valid(const AttrList &model_config) {
    return model_config.i_size() >= 4;
}
static int64 model_config_global_opt_num_cores(const AttrList &model_config) {
    return model_config.i(0);
}
static int64 model_config_this_opt_num_cores(const AttrList &model_config) {
    return model_config.i(1);
}
static int64 model_config_opt_num_infer(const AttrList &model_config) {
    return model_config.i(2);
}
static int64 model_config_timeout(const AttrList &model_config) {
    return model_config.i(3);
}


Status NeuronOp::initialize() {
    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    ch_args.SetMaxSendMessageSize(-1);
    nrtd_address_ = env_get("NEURON_RTD_ADDRESS", "unix:/run/neuron.sock");
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        nrtd_address_, grpc::InsecureChannelCredentials(), ch_args);
    if (nullptr == channel) {
        return errors::Unavailable(
            "cannot establish grpc channel to neuron-rtd server");
    }
    stub_ = nrt::nmgr_v1::NewStub(channel);
    if (nullptr == stub_) {
        return errors::Unavailable("cannot create stub");
    }
    AttrList &model_config = def().attr().at("model_config").list();
    int64_t opt_device_size = -1;
    if (model_config_valid(model_config)) {
        opt_device_size = model_config_global_opt_num_cores(model_config);
    }
    {
        tensorflow::mutex_lock lock(global_neuron_device_manager.global_mutex_);
        if (!global_neuron_device_manager.ready()) {
            TF_RETURN_IF_ERROR(global_neuron_device_manager.initialize(opt_device_size));
#ifdef NEURONTFSERV
            std::signal(SIGINT, sigint_handler);
            std::signal(SIGTERM, sigint_handler);
#endif // NEURONTFSERV
        }
        if (!global_neuron_device_manager.ready()) {
            return errors::FailedPrecondition(
                "global_neuron_device_manager initialization failure");
        }

        // create_eg
        neuron_device_ = global_neuron_device_manager.get_device();
    }

    {
        tensorflow::mutex_lock lock(neuron_device_->mutex_eg_);
        Status status = load(model_config);
        if (!status.ok()) {
            return status;
        }
    }
    VLOG(1) << "load: number of NEFFs: " << neuron_device_->num_executable();

    // check argument sizes
    AttrList &input_names = def().attr().at("input_names").list();
    AttrList &input_dtypes = def().attr().at("input_dtypes").list();
    AttrList &input_shapes = def().attr().at("input_shapes").list();
    AttrList &output_names = def().attr().at("output_names").list();
    AttrList &output_dtypes = def().attr().at("output_dtypes").list();
    AttrList &output_shapes = def().attr().at("output_shapes").list();
    if (input_names.s_size() != input_dtypes.type_size()
            || input_names.s_size() != input_shapes.shape_size()) {
        return errors::FailedPrecondition(
            "incorrect number of inputs: input_names size ", input_names.s_size(),
            ", input_dtypes size ", input_dtypes.type_size(),
            ", input_shapes size ", input_shapes.shape_size());
    }
    if (output_names.s_size() != output_dtypes.type_size()
            || output_names.s_size() != output_shapes.shape_size()) {
        return errors::FailedPrecondition(
            "incorrect number of outputs: output_names size ", output_names.s_size(),
            ", output_dtypes size ", output_dtypes.type_size(),
            ", output_shapes size ", output_shapes.shape_size());
    }
    for (size_t idx = 0; idx < input_dtypes.type_size(); ++idx) {
        Tensor temp_tensor(input_dtypes.type(idx), input_shapes.shape(idx));
        input_tensor_sizes_.push_back(temp_tensor.tensor_data().size());
    }

    // preallocate output tensors (not used by the default infer call)
    std::string nrt_shm_map = env_get("NEURON_RTD_SHM_MAP", "");
    if (!nrt_shm_map.empty()) {
        if (0 == nrtd_address_.find("unix:") && prepare_shared_memory().ok()) {
            use_shared_memory_ = true;
            for (size_t idx = 0; idx < output_dtypes.type_size(); ++idx) {
                output_tensors_.emplace_back(
                    &output_shm_allocs_[idx], output_dtypes.type(idx), output_shapes.shape(idx));
            }
        } else {
            LOG(WARNING) << "shared memory is requested but is not available; "
                         << "using regular grpc for transfering input/output tensors";
            use_shared_memory_ = false;
        }
    }
    AttrList &input_batch_axis = def().attr().at("input_batch_axis").list();
    for (size_t idx = 0; idx < input_batch_axis.i_size(); ++idx) {
        if (-1 != input_batch_axis.i(idx)) {
            enable_dynamic_batch_size_ = true;
            break;
        }
    }
    int64 init_acquire_amount = INFER_SEM_MAX_CAPACITY - (int64)max_num_infers_;
    if (init_acquire_amount >= INFER_SEM_MAX_CAPACITY) {
        LOG(WARNING) << "infer semaphore cannot be correctly initialized;"
                     << " forcing semaphore value to be 1";
        init_acquire_amount = INFER_SEM_MAX_CAPACITY - 1;
    }
    std::string unlimited_threads = env_get("NEURON_UNLIMITED_THREADS", "");
    if (init_acquire_amount > 0 && !infer_sem_initialized_ && "yes" != unlimited_threads) {
        infer_sem_reserve_ptr_ = std::make_shared<xla::Semaphore::ScopedReservation>(
            infer_sem_.ScopedAcquire(init_acquire_amount));
        infer_sem_initialized_ = true;
        int64 infer_sem_capacity = INFER_SEM_MAX_CAPACITY - init_acquire_amount;
        VLOG(1) << "infer semaphore capacity " << infer_sem_capacity;
    }
    ready_ = true;
    return Status::OK();
}

Status NeuronOp::load(const AttrList &model_config) {
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
    request.mutable_h_eg()->set_id(neuron_device_->eg_id());
    WRITE_LOAD_REQUEST;

    // neff_size
    size_t exec_total_size = def().attr().at("executable").s().size();
    request.set_neff_size(exec_total_size);
    WRITE_LOAD_REQUEST;

    // model_params
    uint32_t uint32_timeout = 10;
    if (model_config_valid(model_config)) {
        int64 int64_timeout = model_config_timeout(model_config);
        if (0 < int64_timeout) {
            uint32_timeout = (uint32_t)int64_timeout;
        }
    } else {
        uint32_timeout = 10;
    }
    std::string infer_timeout_str = env_get("NEURON_INFER_TIMEOUT_SEC", "10");
    int int_timeout = stoi_no_throw(infer_timeout_str);
    if (int_timeout < 0) {
        LOG(WARNING) << "NEURON_INFER_TIMEOUT_SEC=" << infer_timeout_str
                     << " is invalid; using default value " << uint32_timeout
                     << " seconds.";
    } else {
        uint32_timeout = (uint32_t)int_timeout;
    }
    max_num_infers_ = DEFAULT_MAX_NUM_INFER;
    if (model_config_valid(model_config)) {
        int64 opt_num_infer = model_config_opt_num_infer(model_config);
        if (opt_num_infer > 0) {
            opt_num_infer += NRTD_NUM_CPU_THREADS;  // add some extras for CPU nodes
            max_num_infers_ = (uint32_t)opt_num_infer;
        }
    }
    max_num_infers_ = max_num_infers_ > 1 ? max_num_infers_ : 1;
    std::string ninfer_str = env_get("NEURON_MAX_NUM_INFERS", "");
    if (!ninfer_str.empty()) {
        int64 int_ninfer = (int64)stoi_no_throw(ninfer_str);
        if (int_ninfer < 1 || int_ninfer > INFER_SEM_MAX_CAPACITY) {
            LOG(WARNING) << "NEURON_MAX_NUM_INFERS=" << ninfer_str
                         << " is invalid; using default value "
                         << max_num_infers_  << ".";
        } else {
            max_num_infers_ = (uint32_t)int_ninfer;
        }
    }
    if (model_config_valid(model_config)) {
        // enforce max_num_infers_ = 1 if ncg size is insufficient
        int64 int64_opt_num_cores = model_config_this_opt_num_cores(model_config);
        if (int64_opt_num_cores < NeuronDeviceManager::MIN_NUM_CORES
                || int64_opt_num_cores > NeuronDeviceManager::MAX_NUM_CORES) {
            max_num_infers_ = NRTD_INSUFFICIENT_NUM_INFER;
        } else {
            uint32_t opt_num_cores = (uint32_t)int64_opt_num_cores;
            if (neuron_device_->num_cores() < opt_num_cores) {
                max_num_infers_ = NRTD_INSUFFICIENT_NUM_INFER;
            }
        }
    }
    nrt::model_params *model_params = request.mutable_model_params();
    model_params->mutable_timeout()->set_data(uint32_timeout);
    model_params->mutable_ninfer()->set_data(max_num_infers_ + 1);
    WRITE_LOAD_REQUEST;

    // neff file content
    StringPiece executable_view(def().attr().at("executable").s());
    for (size_t pos = 0; pos < exec_total_size; pos += EXEC_MAX_CHUNK_SIZE) {
        size_t remaining = exec_total_size - pos;
        size_t chunk_size = std::min(remaining, EXEC_MAX_CHUNK_SIZE);
        StringPiece file_chunk = executable_view.substr(pos, chunk_size);
        request.mutable_neff_chunk()->set_chunk(file_chunk.data(), chunk_size);
        WRITE_LOAD_REQUEST;
    }
    if (!writer->WritesDone()) {
        return errors::Internal("neuron-rtd load failure - broken stream");
    }
    grpc::Status status = writer->Finish();
    NRT_CHECK_RETURN("load", status, response);
    nn_id_ = response.h_nn().id();
    load_done_ = true;
    neuron_device_->register_executable(nn_id_);
    return Status::OK();
}

Status NeuronOp::prepare_shared_memory() {
    for (size_t idx = 0; idx < input_tensor_sizes_.size(); ++idx) {
        size_t shm_size = input_tensor_sizes_[idx];
        input_shms_.emplace_back(shm_size);
        TF_RETURN_IF_ERROR(input_shms_.back().initialize(stub_, nn_id_));
        VLOG(1) << "input shared memory " << input_shms_.back().name()
                << " ready at address " << input_shms_.back().ptr();
    }
    AttrList &output_dtypes = def().attr().at("output_dtypes").list();
    AttrList &output_shapes = def().attr().at("output_shapes").list();
    for (size_t idx = 0; idx < output_dtypes.type_size(); ++idx) {
        Tensor temp_tensor(output_dtypes.type(idx), output_shapes.shape(idx));
        size_t shm_size = temp_tensor.tensor_data().size();
        output_shms_.emplace_back(shm_size);
        TF_RETURN_IF_ERROR(output_shms_.back().initialize(stub_, nn_id_));
        VLOG(1) << "output shared memory " << output_shms_.back().name()
                << " ready at address " << output_shms_.back().ptr();
    }
    for (auto &out_shm : output_shms_) {
        output_shm_allocs_.emplace_back(&out_shm);
    }
    return Status::OK();
}


NeuronOp::~NeuronOp() {
    VLOG(1) << "calling NeuronOp destructor";
    if (!unloaded_) {
        tensorflow::mutex_lock lock(mutex_model_);
        if (nullptr == neuron_device_) {
            VLOG(1) << "neuron_device_ not available; not tearing down";
            return;
        }

        {
            tensorflow::mutex_lock lock(neuron_device_->mutex_eg_);
            // stop
            if (neuron_device_->running(nn_id_)) {
                grpc::ClientContext context;
                nrt::stop_request request;
                request.mutable_h_nn()->set_id(nn_id_);
                nrt::stop_response response;
                grpc::Status status = stub_->stop(&context, request, &response);
                NRT_CHECK_LOG("stop", status, response);
                neuron_device_->set_running(NRT_INVALID_NN_ID);
            }

            // unload
            if (load_done_) {
                grpc::ClientContext context;
                nrt::unload_request request;
                request.mutable_h_nn()->set_id(nn_id_);
                nrt::unload_response response;
                grpc::Status status = stub_->unload(&context, request, &response);
                NRT_CHECK_LOG("unload", status, response);
            }
            neuron_device_->deregister_executable(nn_id_);
            VLOG(1) << "unload: number of NEFFs: " << neuron_device_->num_executable();
            VLOG(1) << "unload from NeuronOp::~NeuronOp";

            // unmap all shared memories
            for (auto &shm : input_shms_) {
                shm.clear(stub_);
            }
            for (auto &shm : output_shms_) {
                shm.clear(stub_);
            }
        }

        {
            tensorflow::mutex_lock lock(global_neuron_device_manager.global_mutex_);
            // clear global_neuron_device_manager if it's empty
            if (global_neuron_device_manager.is_empty()) {
                global_neuron_device_manager.clear();
            }
        }
        VLOG(1) << "NeuronOp destructor done";
        ready_ = false;
        unloaded_ = true;
    }
}


static Status tensor_memcpy(Tensor *tensor, StringPiece &source, int64 memcpy_size=-1) {
    if (memcpy_size < 0 ? source.size() != tensor->tensor_data().size()
                        : memcpy_size > tensor->tensor_data().size()) {
        return errors::OutOfRange(
            "unexpected tensor size in tensor_memcpy, source size: ",
            source.size(), ", target size: ", tensor->tensor_data().size());
    }
    if (memcpy_size < 0) {
        memcpy_size = source.size();
    }
    #define CASE_MEMCPY_TENSOR(TF_DataType, TTYPE) {            \
        case (TF_DataType):                                     \
            std::memcpy(tensor->unaligned_flat<TTYPE>().data(), \
                        source.data(), memcpy_size);            \
        break;                                                  \
    }
    switch (tensor->dtype()) {
        CASE_MEMCPY_TENSOR(DT_HALF,     Eigen::half);
        CASE_MEMCPY_TENSOR(DT_BFLOAT16, tensorflow::bfloat16);
        CASE_MEMCPY_TENSOR(DT_FLOAT,    float);
        CASE_MEMCPY_TENSOR(DT_UINT8,    tensorflow::uint8);
        CASE_MEMCPY_TENSOR(DT_INT8,     tensorflow::int8);
        CASE_MEMCPY_TENSOR(DT_UINT16,   tensorflow::uint16);
        CASE_MEMCPY_TENSOR(DT_INT16,    tensorflow::int16);
        CASE_MEMCPY_TENSOR(DT_UINT32,   tensorflow::uint32);
        CASE_MEMCPY_TENSOR(DT_INT32,    tensorflow::int32);
        CASE_MEMCPY_TENSOR(DT_QUINT8,   tensorflow::quint8);
        CASE_MEMCPY_TENSOR(DT_QUINT16,  tensorflow::quint16);
        CASE_MEMCPY_TENSOR(DT_QINT32,   tensorflow::qint32);
    default:
        return errors::InvalidArgument("tensor->dtype() is unsupported");
    }
    return Status::OK();
}


static Status tensor_memset(Tensor *tensor, int ch) {
    #define CASE_MEMSET_TENSOR(TF_DataType, TTYPE) {            \
        case (TF_DataType):                                     \
            std::memset(tensor->unaligned_flat<TTYPE>().data(), \
                        ch, tensor->tensor_data().size());      \
        break;                                                  \
    }
    switch (tensor->dtype()) {
        CASE_MEMSET_TENSOR(DT_HALF,     Eigen::half);
        CASE_MEMSET_TENSOR(DT_BFLOAT16, tensorflow::bfloat16);
        CASE_MEMSET_TENSOR(DT_FLOAT,    float);
        CASE_MEMSET_TENSOR(DT_UINT8,    tensorflow::uint8);
        CASE_MEMSET_TENSOR(DT_INT8,     tensorflow::int8);
        CASE_MEMSET_TENSOR(DT_UINT16,   tensorflow::uint16);
        CASE_MEMSET_TENSOR(DT_INT16,    tensorflow::int16);
        CASE_MEMSET_TENSOR(DT_UINT32,   tensorflow::uint32);
        CASE_MEMSET_TENSOR(DT_INT32,    tensorflow::int32);
        CASE_MEMSET_TENSOR(DT_QUINT8,   tensorflow::quint8);
        CASE_MEMSET_TENSOR(DT_QUINT16,  tensorflow::quint16);
        CASE_MEMSET_TENSOR(DT_QINT32,   tensorflow::qint32);
    default:
        return errors::InvalidArgument("tensor->dtype() is unsupported");
    }
    return Status::OK();
}


void NeuronOp::Compute(OpKernelContext *ctx) {
    FALTimestamps timestamps;
    timestamps.mark_enter();

    if (!ready_) {
        tensorflow::mutex_lock lock(mutex_model_);
        if (!ready_) {
            OP_REQUIRES_OK(ctx, initialize());
        }
    }

    AttrList &input_names = def().attr().at("input_names").list();
    AttrList &output_shapes = def().attr().at("output_shapes").list();
    std::vector<const Tensor*> input_tensors;
    for (auto idx = 0; idx < ctx->num_inputs(); ++idx) {
        input_tensors.push_back(&ctx->input(idx));
    }
    OP_REQUIRES(ctx, input_tensors.size() == input_names.s_size(),
                errors::InvalidArgument("incorrect number of input tensors"));

    int64_t batch_size = UNINIT_BATCH_SIZE;
    int64_t k_batch_size = UNINIT_BATCH_SIZE;
    std::vector<bool> is_batch_input_tensors;
    std::vector<bool> is_batch_output_tensors;
    bool use_dynamic_batch_size = false;
    AttrList &output_names = def().attr().at("output_names").list();
    AttrList &input_batch_axis = def().attr().at("input_batch_axis").list();
    AttrList &output_batch_axis = def().attr().at("output_batch_axis").list();
    if (enable_dynamic_batch_size_ && input_names.s_size() == input_batch_axis.i_size() &&
            output_names.s_size() == output_batch_axis.i_size()) {
        AttrList &input_shapes = def().attr().at("input_shapes").list();
        for (auto idx = 0; idx < input_tensors.size(); ++idx) {
            bool is_batch_tensor = false;
            const Tensor *tptr = input_tensors[idx];
            TensorShape shape(tptr->shape());
            TensorShape k_shape(input_shapes.shape(idx));
            if (0 == input_batch_axis.i(idx)) {
                OP_REQUIRES(ctx, shape.dims() > 0,
                            errors::InvalidArgument(
                                "no batch-dimension found on input tensor ",
                                input_names.s(idx), " with shape ", shape.DebugString()));
                if (UNINIT_BATCH_SIZE == batch_size) {
                    batch_size = shape.dim_size(0);
                    k_batch_size = k_shape.dim_size(0);
                    OP_REQUIRES(ctx, batch_size > 0,
                                errors::Internal(
                                    "incorrect internal batch size inferred from input tensor ",
                                    input_names.s(idx), " with shape ", shape.DebugString()));
                } else {
                    OP_REQUIRES(ctx, batch_size == shape.dim_size(0),
                                errors::InvalidArgument(
                                    "incorrect batch size found on input tensor ",
                                    input_names.s(idx), ", tensor shape ", shape.DebugString(),
                                    ", internal batch size ", batch_size));
                }
                shape.RemoveDim(0);
                k_shape.RemoveDim(0);
                is_batch_tensor = batch_size != k_batch_size;
                use_dynamic_batch_size = is_batch_tensor;
            }
            OP_REQUIRES(ctx, shape == k_shape,
                        errors::InvalidArgument(
                            "incorrect shape found on input tensor ", input_names.s(idx),
                            ", inference time shape ", tptr->shape().DebugString(),
                            ", expected shape ", input_shapes.shape(idx).DebugString()));
            is_batch_input_tensors.push_back(is_batch_tensor);
        }
        for (auto idx = 0; idx < output_names.s_size(); ++idx) {
            bool is_batch_tensor = false;
            if (0 == output_batch_axis.i(idx)) {
                TensorShape k_shape(output_shapes.shape(idx));
                OP_REQUIRES(ctx, k_shape.dims() > 0,
                            errors::InvalidArgument(
                                "no batch-dimension found on output tensor ",
                                output_names.s(idx), " with Neuron shape ",
                                k_shape.DebugString()));
                OP_REQUIRES(ctx, k_batch_size == k_shape.dim_size(0),
                            errors::InvalidArgument(
                                "incorrect batch size found on output tensor ",
                                output_names.s(idx), ", Neuron tensor shape ",
                                k_shape.DebugString(), ", Neuron batch size ",
                                k_batch_size));
                is_batch_tensor = batch_size != k_shape.dim_size(0);
            }
            is_batch_output_tensors.push_back(is_batch_tensor);
        }
    }
    OP_REQUIRES(ctx, ctx->num_outputs() == output_names.s_size(),
                errors::InvalidArgument("incorrect number of output tensors"));

    if (use_dynamic_batch_size) {
        int64_t pad_batch_size = ((batch_size - 1) / k_batch_size + 1) * k_batch_size;
        std::vector<Tensor*> batch_output_tensors;
        for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
            Tensor *batch_out_tensor = nullptr;
            TensorShape shape(output_shapes.shape(idx));
            if (is_batch_output_tensors[idx]) {
                shape.set_dim(0, batch_size);
            }
            OP_REQUIRES_OK(ctx, ctx->allocate_output(idx, shape, &batch_out_tensor));
            batch_output_tensors.push_back(batch_out_tensor);
        }

        std::vector<std::vector<Tensor> > batches_neuron_input_tensors;
        int64_t num_batches = pad_batch_size / k_batch_size;
        for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int64_t dim0_start = batch_idx * k_batch_size;
            int64_t dim0_limit = batch_idx * k_batch_size + k_batch_size;
            batches_neuron_input_tensors.emplace_back();
            for (auto idx = 0; idx < input_tensors.size(); ++idx) {
                if (is_batch_input_tensors[idx]) {
                    if (batch_idx == num_batches - 1) {
                        TensorShape ps_shape(input_tensors[idx]->shape());
                        ps_shape.set_dim(0, k_batch_size);
                        Tensor pad_end_slice(input_tensors[idx]->dtype(), ps_shape);
                        Tensor zero_slice = pad_end_slice.Slice(
                            k_batch_size - (pad_batch_size - batch_size), k_batch_size);
                        OP_REQUIRES_OK(ctx, tensor_memset(&zero_slice, 0));
                        Tensor end_slice = input_tensors[idx]->Slice(
                            dim0_start, batch_size);
                        StringPiece t_data = end_slice.tensor_data();
                        OP_REQUIRES_OK(ctx, tensor_memcpy(
                            &pad_end_slice, t_data, t_data.size()));
                        batches_neuron_input_tensors.back().emplace_back(pad_end_slice);
                    } else {
                        batches_neuron_input_tensors.back().emplace_back(
                            input_tensors[idx]->Slice(dim0_start, dim0_limit));
                    }
                } else {
                    batches_neuron_input_tensors.back().emplace_back();
                }
            }
        }

        int64_t window_size = max_num_infers_ > 1 ? max_num_infers_ : 1;
        window_size = std::min(window_size, num_batches);
        std::queue<uint64_t> cookie_queue;
        std::queue<int64_t> batch_idx_queue;
        int64_t post_bidx = 0;
        {   // scope of semaphore reservation queue
            std::queue<xla::Semaphore::ScopedReservation> reservation_queue;
            AttrList &output_dtypes = def().attr().at("output_dtypes").list();
            {   // lock EG; we assume this op instance is not loaded into more than
                // one EG and so we just use this EG's lock
                tensorflow::mutex_lock lock(neuron_device_->mutex_eg_);
                OP_REQUIRES_OK(ctx, neuron_device_->is_valid());
                OP_REQUIRES_OK(ctx, start_model());
                timestamps.mark_above_nrtd_infer();
                // post ninfer ones
                for (post_bidx = 0; post_bidx < window_size; ++post_bidx) {
                    std::vector<const Tensor*> sliced_inputs;
                    for (auto idx = 0; idx < input_names.s_size(); ++idx) {
                        if (is_batch_input_tensors[idx]) {
                            sliced_inputs.push_back(
                                &batches_neuron_input_tensors[post_bidx][idx]);
                        } else {
                            sliced_inputs.push_back(input_tensors[idx]);
                        }
                    }
                    uint64_t post_cookie;
                    reservation_queue.push(infer_sem_.ScopedAcquire(1));
                    OP_REQUIRES_OK(ctx, infer_post(&post_cookie, sliced_inputs));
                    cookie_queue.push(post_cookie);
                    batch_idx_queue.push(post_bidx);
                }

                // wait one and post one
                for (post_bidx = window_size; post_bidx < num_batches; ++post_bidx) {
                    std::vector<const Tensor*> sliced_inputs;
                    for (auto idx = 0; idx < input_names.s_size(); ++idx) {
                        if (is_batch_input_tensors[idx]) {
                            sliced_inputs.push_back(
                                &batches_neuron_input_tensors[post_bidx][idx]);
                        } else {
                            sliced_inputs.push_back(input_tensors[idx]);
                        }
                    }

                    // wait one
                    std::vector<Tensor> temp_outputs;
                    for (auto idx = 0; idx < output_dtypes.type_size(); ++idx) {
                        temp_outputs.emplace_back();
                        if (is_batch_output_tensors[idx]) {
                            OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                                output_dtypes.type(idx), output_shapes.shape(idx),
                                &temp_outputs[idx]));
                        }
                    }
                    std::vector<Tensor*> output_tensors;
                    for (auto idx = 0; idx < temp_outputs.size(); ++idx) {
                        Tensor *out_tensor = is_batch_output_tensors[idx] ?
                            &temp_outputs[idx] : batch_output_tensors[idx];
                        output_tensors.push_back(out_tensor);
                    }
                    uint64_t wait_cookie = cookie_queue.front();
                    int64_t wait_batch_idx = batch_idx_queue.front();
                    cookie_queue.pop();
                    batch_idx_queue.pop();
                    int64_t dim0_start = wait_batch_idx * k_batch_size;
                    int64_t dim0_limit = wait_batch_idx * k_batch_size + k_batch_size;
                    nrt::infer_response response;
                    OP_REQUIRES_OK(ctx, infer_wait(&response, wait_cookie));
                    OP_REQUIRES_OK(ctx, copy_output_tensors(&output_tensors, response));
                    for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
                        if (is_batch_output_tensors[idx]) {
                            StringPiece k_data = temp_outputs[idx].tensor_data();
                            Tensor slice = batch_output_tensors[idx]->Slice(
                                dim0_start, std::min(dim0_limit, batch_size));
                            OP_REQUIRES_OK(ctx, tensor_memcpy(
                                &slice, k_data, slice.tensor_data().size()));
                        }
                    }

                    // post next one
                    uint64_t post_cookie;
                    OP_REQUIRES_OK(ctx, infer_post(&post_cookie, sliced_inputs));
                    cookie_queue.push(post_cookie);
                    batch_idx_queue.push(post_bidx);
                }
            }   // unlock EG

            // wait for remaining ones in the queue
            for (int64_t dummy_idx = 0; dummy_idx < num_batches; ++dummy_idx) {
                if (cookie_queue.empty() || batch_idx_queue.empty()) {
                    break;
                }
                std::vector<Tensor> temp_outputs;
                for (auto idx = 0; idx < output_dtypes.type_size(); ++idx) {
                    temp_outputs.emplace_back();
                    if (is_batch_output_tensors[idx]) {
                        OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            output_dtypes.type(idx), output_shapes.shape(idx),
                            &temp_outputs[idx]));
                    }
                }
                std::vector<Tensor*> output_tensors;
                for (auto idx = 0; idx < temp_outputs.size(); ++idx) {
                    Tensor *out_tensor = is_batch_output_tensors[idx] ?
                        &temp_outputs[idx] : batch_output_tensors[idx];
                    output_tensors.push_back(out_tensor);
                }
                uint64_t wait_cookie = cookie_queue.front();
                int64_t wait_batch_idx = batch_idx_queue.front();
                cookie_queue.pop();
                batch_idx_queue.pop();
                int64_t dim0_start = wait_batch_idx * k_batch_size;
                int64_t dim0_limit = wait_batch_idx * k_batch_size + k_batch_size;
                nrt::infer_response response;
                OP_REQUIRES_OK(ctx, infer_wait(&response, wait_cookie));
                reservation_queue.pop();
                OP_REQUIRES_OK(ctx, copy_output_tensors(&output_tensors, response));
                for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
                    if (is_batch_output_tensors[idx]) {
                        StringPiece k_data = temp_outputs[idx].tensor_data();
                        Tensor slice = batch_output_tensors[idx]->Slice(
                            dim0_start, std::min(dim0_limit, batch_size));
                        OP_REQUIRES_OK(ctx, tensor_memcpy(
                            &slice, k_data, slice.tensor_data().size()));
                    }
                }
            }
            timestamps.mark_below_nrtd_infer();
        }   // semaphore reservation queue goes out of scope
        timestamps.mark_exit();
        VLOG(1) << timestamps.timing_string();
    } else if (profile_enabled_ || use_shared_memory_) {
        VLOG(1) << "profile/shm enabled -- lock stop/start/infer altogether";
        std::vector<Tensor*> output_tensors;
        for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
            output_tensors.emplace_back();
            OP_REQUIRES_OK(ctx, ctx->allocate_output(idx, output_shapes.shape(idx),
                                                     &output_tensors[idx]));
        }
        {   // lock EG; we assume this op instance is not loaded into more than
            // one EG and so we just use this EG's lock
            tensorflow::mutex_lock lock(neuron_device_->mutex_eg_);
            OP_REQUIRES_OK(ctx, neuron_device_->is_valid());
            OP_REQUIRES_OK(ctx, start_model());
            profile_start_session();
            Status status = infer(&output_tensors, input_tensors, &timestamps);
            profile_stop_session();
            OP_REQUIRES_OK(ctx, status);
        }   // unlock EG
        timestamps.mark_exit();
        VLOG(1) << timestamps.timing_string();
    } else {
        std::vector<Tensor*> output_tensors;
        nrt::infer_response response;
        {
            std::queue<xla::Semaphore::ScopedReservation> reservation_queue;
            uint64_t cookie;
            {   // lock EG; we assume this op instance is not loaded into more than
                // one EG and so we just use this EG's lock
                tensorflow::mutex_lock lock(neuron_device_->mutex_eg_);
                OP_REQUIRES_OK(ctx, neuron_device_->is_valid());
                OP_REQUIRES_OK(ctx, start_model());
                reservation_queue.push(infer_sem_.ScopedAcquire(1));
                timestamps.mark_above_nrtd_infer();
                OP_REQUIRES_OK(ctx, infer_post(&cookie, input_tensors));
            }   // unlock EG
            for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
                output_tensors.emplace_back();
                OP_REQUIRES_OK(ctx, ctx->allocate_output(idx, output_shapes.shape(idx),
                                                         &output_tensors[idx]));
            }
            OP_REQUIRES_OK(ctx, infer_wait(&response, cookie));
            reservation_queue.pop();
            timestamps.mark_below_nrtd_infer();
        }
        OP_REQUIRES_OK(ctx, copy_output_tensors(&output_tensors, response));
        timestamps.mark_exit();
        VLOG(1) << timestamps.timing_string();
    }
}


Status NeuronOp::start_model() {
    grpc::Status status;
    if (!neuron_device_->running(nn_id_) && neuron_device_->is_busy()) {
        // if nn_id_ is not running, stop the current running model
        grpc::ClientContext context;
        nrt::stop_request request;
        request.mutable_h_nn()->set_id(neuron_device_->nn_get_current_running());
        nrt::stop_response response;
        status = stub_->stop(&context, request, &response);
        NRT_CHECK_RETURN("stop", status, response);
        neuron_device_->set_running(NRT_INVALID_NN_ID);
    }
    if (!neuron_device_->is_busy()) {
        // if no model is running, start nn_id_
        grpc::ClientContext context;
        nrt::start_request request;
        request.mutable_h_nn()->set_id(nn_id_);
        nrt::start_response response;
        status = stub_->start(&context, request, &response);
        NRT_CHECK_RETURN("start", status, response);
        neuron_device_->set_running(nn_id_);
    }
    return Status::OK();
}


template<typename... Args>
Status subprocess_run(Args... args) {
    pid_t fork_pid;
    SYS_FAIL_RETURN((fork_pid = fork()) < 0, "fork");
    if (0 == fork_pid) {
        execlp(args..., (char*)NULL);
        LOG(ERROR) << "execlp failed";
        _exit(1);
    } else {
        int status;
        SYS_FAIL_RETURN(waitpid(fork_pid, &status, 0) < 0, "waitpid");
        if (!(WIFEXITED(status) && 0 == WEXITSTATUS(status))) {
            return errors::Internal("child process did not exit gracefully");
        }
    }
    return Status::OK();
}


static std::string mangle_op_name(const std::string &op_name) {
    std::string new_op_name(op_name);
    for (size_t idx = 0; idx < new_op_name.length(); ++idx) {
        if ('/' == new_op_name[idx]) {
            new_op_name[idx] = '+';
        }
    }
    return new_op_name;
}


void NeuronOp::profile_dump_info() {
    std::string filename_base = profile_dir_ + "/" + mangle_op_name(def().name());
    std::string filename_pb = filename_base + ".pb";
    std::string filename_neff = filename_base + ".neff";
    std::ofstream(filename_pb, std::ios::binary) << def().attr().at("graph_def").s();
    std::ofstream(filename_neff, std::ios::binary) << def().attr().at("executable").s();
}


void NeuronOp::profile_start_session() {
    if (profile_enabled_) {
        std::string new_op_name = mangle_op_name(def().name());
        std::ostringstream filename_stream;
        filename_stream << profile_dir_ << "/" << new_op_name << "-"
                        << nn_id_ << "-" << profile_session_id_ << ".ntff";
        profile_session_filename_ = filename_stream.str();
        std::ostringstream cmd_stream;
        cmd_stream << "neuron-profile start-session -s " << profile_session_filename_
                   << " -a " << nrtd_address_ << " " << nn_id_;
        VLOG(1) << "Starting profiling session by " << cmd_stream.str();
        std::ostringstream nn_id_stream;
        nn_id_stream << nn_id_;
        Status status = subprocess_run(
            "neuron-profile", "neuron-profile", "start-session", "-s",
            profile_session_filename_.c_str(), "-a", nrtd_address_.c_str(),
            nn_id_stream.str().c_str());
        if (!status.ok()) {
            profile_session_filename_ = "";
            LOG(WARNING) << "neuron-profile start-session failed. "
                         << "Did you install aws-neuron-tools-core?";
            return;
        }
        profile_session_id_++;
    }
}


void NeuronOp::profile_stop_session() {
    if (profile_enabled_ && !profile_session_filename_.empty()) {
        std::ostringstream cmd_stream;
        cmd_stream << "neuron-profile stop-session -s " << profile_session_filename_;
        VLOG(1) << "Stopping profiling session by " << cmd_stream.str();
        Status status = subprocess_run(
            "neuron-profile", "neuron-profile", "stop-session", "-s",
            profile_session_filename_.c_str());
        if (!status.ok()) {
            LOG(ERROR) << "neuron-profile stop-session failed";
        }
        profile_session_filename_ = "";
    }
}


Status NeuronOp::infer(std::vector<Tensor*> *output_tensors,
                       const std::vector<const Tensor*> &input_tensors,
                       FALTimestamps *timestamps) {
    if (!ready_) {
        return errors::FailedPrecondition("not ready for inference");
    }

    // set input tensors
    AttrList &input_names = def().attr().at("input_names").list();
    AttrList &output_names = def().attr().at("output_names").list();
    if (input_tensors.size() != input_names.s_size()) {
        return errors::Internal(
            "incorrect number of input tensors, input_tensors size ",
            input_tensors.size(), ", input_names size", input_names.s_size());
    }
    nrt::infer_request request;
    for (size_t idx = 0; idx < input_names.s_size(); ++idx) {
        nrt::infer_io *infer_io = request.add_ifmap();
        infer_io->set_name(input_names.s(idx));
        StringPiece tensor_data(input_tensors[idx]->tensor_data());
        if (tensor_data.size() != input_tensor_sizes_[idx]) {
            return errors::Internal(
                "incorrect input tensor size ", tensor_data.size(), " found on ",
                input_names.s(idx), " (", input_tensor_sizes_[idx], ")");
        }
        if (use_shared_memory_) {
            infer_io->mutable_buf_shm()->set_path(input_shms_[idx].name());
            std::memcpy(input_shms_[idx].ptr(), tensor_data.data(), tensor_data.size());
        } else {
            infer_io->set_buf(tensor_data.data(), tensor_data.size());
        }
    }
    if (use_shared_memory_) {
        for (size_t idx = 0; idx < output_names.s_size(); ++idx) {
            nrt::infer_io *infer_io = request.add_shm_ofmap();
            infer_io->set_name(output_names.s(idx));
            infer_io->mutable_buf_shm()->set_path(output_shms_[idx].name());
        }
    }
    request.mutable_h_nn()->set_id(nn_id_);

    // infer
    grpc::ClientContext context;
    nrt::infer_response response;
    timestamps->mark_above_nrtd_infer();
    grpc::Status status = stub_->infer(&context, request, &response);
    timestamps->mark_below_nrtd_infer();
    if (status.ok()) {
        // ignore inf/nan errors
        if (nrt::nerr::NERR_INFER_COMPLETED_WITH_NUM_ERR == response.status().code()) {
            response.mutable_status()->set_code(nrt::nerr::NERR_OK);
        }
    }
    NRT_CHECK_RETURN("infer", status, response);

    if (use_shared_memory_) {
        for (size_t idx = 0; idx < output_names.s_size(); ++idx) {
            StringPiece out_tensor_raw(output_tensors_[idx].tensor_data());
            Tensor *out_tensor = output_tensors->at(idx);
            TF_RETURN_WITH_CONTEXT_IF_ERROR(tensor_memcpy(out_tensor, out_tensor_raw),
                                            "tensor_memcpy failure on tensor name: ",
                                            output_names.s(idx));
        }
        return Status::OK();
    }

    // set output tensors
    std::vector<StringPiece> raw_output_tensors;
    std::unordered_map<std::string, StringPiece> map_name_raw;
    for (const auto &infer_io : response.ofmap()) {
        map_name_raw.emplace(infer_io.name(), infer_io.buf());
    }
    for (size_t idx = 0; idx < output_names.s_size(); ++idx) {
        if (map_name_raw.find(output_names.s(idx)) == map_name_raw.end()) {
            return errors::Internal(
                "tensor name", output_names.s(idx),
                " not found in infer_response.ofmap()");
        }
        raw_output_tensors.push_back(map_name_raw[output_names.s(idx)]);
    }
    for (size_t idx = 0; idx < output_names.s_size(); ++idx) {
        StringPiece out_tensor_raw = raw_output_tensors[idx];
        Tensor *out_tensor = output_tensors->at(idx);
        TF_RETURN_WITH_CONTEXT_IF_ERROR(tensor_memcpy(out_tensor, out_tensor_raw),
                                        "tensor_memcpy failure on tensor name: ",
                                        output_names.s(idx));
    }

    return Status::OK();
}


Status NeuronOp::infer_post(uint64_t *infer_post_cookie,
                            const std::vector<const Tensor*> &input_tensors) {
    if (!ready_) {
        return errors::FailedPrecondition("not ready for inference");
    }

    // set input tensors
    AttrList &input_names = def().attr().at("input_names").list();
    if (input_tensors.size() != input_names.s_size()) {
        return errors::Internal(
            "incorrect number of input tensors, input_tensors size ",
            input_tensors.size(), ", input_names size", input_names.s_size());
    }

    nrt::infer_request request;
    for (size_t idx = 0; idx < input_names.s_size(); ++idx) {
        nrt::infer_io *infer_io = request.add_ifmap();
        infer_io->set_name(input_names.s(idx));
        StringPiece tensor_data(input_tensors[idx]->tensor_data());
        if (tensor_data.size() != input_tensor_sizes_[idx]) {
            return errors::Internal(
                "incorrect input tensor size ", tensor_data.size(), " found on ",
                input_names.s(idx), " (", input_tensor_sizes_[idx], ")");
        }
        void *data = (void*)tensor_data.data();
        infer_io->set_buf(data, tensor_data.size());
    }
    request.mutable_h_nn()->set_id(nn_id_);

    // infer
    grpc::ClientContext context;
    nrt::infer_post_response response;
    grpc::Status status = stub_->infer_post(&context, request, &response);
    NRT_CHECK_RETURN("infer_post", status, response);
    *infer_post_cookie = response.cookie();
    return Status::OK();
}


Status NeuronOp::infer_wait(nrt::infer_response *response,
                            uint64_t infer_post_cookie) {
    if (!ready_) {
        return errors::FailedPrecondition("not ready for inference");
    }

    nrt::infer_wait_request request;
    request.set_cookie(infer_post_cookie);

    // infer_wait
    grpc::ClientContext context;
    grpc::Status status = stub_->infer_wait(&context, request, response);
    if (status.ok()) {
        // ignore inf/nan errors
        if (nrt::nerr::NERR_INFER_COMPLETED_WITH_NUM_ERR == response->status().code()) {
            response->mutable_status()->set_code(nrt::nerr::NERR_OK);
        }
    }
    NRT_CHECK_RETURN("infer_wait", status, *response);
    return Status::OK();
}


Status NeuronOp::copy_output_tensors(std::vector<Tensor*> *output_tensors,
                                     const nrt::infer_response &response) {
    // set output tensors
    std::vector<StringPiece> raw_output_tensors;
    std::unordered_map<std::string, StringPiece> map_name_raw;
    for (const auto &infer_io : response.ofmap()) {
        map_name_raw.emplace(infer_io.name(), infer_io.buf());
    }
    AttrList &output_names = def().attr().at("output_names").list();
    for (size_t idx = 0; idx < output_names.s_size(); ++idx) {
        if (map_name_raw.find(output_names.s(idx)) == map_name_raw.end()) {
            return errors::Internal("tensor name", output_names.s(idx),
                                    " not found in infer_response.ofmap()");
        }
        raw_output_tensors.push_back(map_name_raw[output_names.s(idx)]);
    }
    for (size_t idx = 0; idx < output_names.s_size(); ++idx) {
        StringPiece out_tensor_raw = raw_output_tensors[idx];
        Tensor *out_tensor = output_tensors->at(idx);
        TF_RETURN_WITH_CONTEXT_IF_ERROR(tensor_memcpy(out_tensor, out_tensor_raw),
                                        "tensor_memcpy failure on tensor name: ",
                                        output_names.s(idx));
    }
    return Status::OK();
}


REGISTER_KERNEL_BUILDER(Name("NeuronOp").Device(DEVICE_CPU), NeuronOp);

}  // namespace neuron
}  // namespace tensorflow

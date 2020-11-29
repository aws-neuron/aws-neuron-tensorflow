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

#include "../runtime_io.h"
#include "./neuron_op.h"


namespace tensorflow {


namespace register_kernel {
class NeuronName {
public:
    const KernelDef *Build() {
        // Need to allocate from an existing one to workaround the bug that setting one attribute
        // can affect other attributes
        KernelDef *kernel = new KernelDef(GetRegisteredKernelsForOp("IdentityN").kernel(0));
        kernel->clear_op();
        kernel->clear_device_type();
        kernel->clear_constraint();
        kernel->clear_host_memory_arg();
        kernel->clear_label();
        kernel->clear_priority();
        kernel->set_op(GetRegisteredKernelsForOp("NeuronOp").kernel_size() ? "_no_register" : "NeuronOp");
        kernel->set_device_type(DEVICE_CPU);
        return kernel;
    }
};
}  // namespace register_kernel


namespace neuron {


static const int64 UNINIT_BATCH_SIZE = -8;  // magic number for uninitialized batch size
extern NeuronDeviceManager global_neuron_device_manager;


class NeuronModelConfig {
public:
    void parse_opt_device_size(AttrList &model_config) {
        if (model_config_valid(model_config)) {
            opt_device_size_ = model_config_global_opt_num_cores(model_config);
            max_num_duplicates_ = model_config_max_num_duplicates(model_config);
        }
    }

    void parse_timeout(AttrList &model_config) {
        if (model_config_valid(model_config)) {
            int64 int64_timeout = model_config_timeout(model_config);
            if (int64_timeout > 0) {
                timeout_ = (uint32_t)int64_timeout;
                return;
            }
        }
        timeout_ = 10;
        std::string infer_timeout_str = env_get("NEURON_FRAMEWORK_INFER_TIMEOUT_SEC", "10");
        int int_timeout = stoi_no_throw(infer_timeout_str);
        if (int_timeout <= 0) {
            LOG(WARNING) << "NEURON_FRAMEWORK_INFER_TIMEOUT_SEC=" << infer_timeout_str
                         << " is invalid; using default value " << timeout_ << " seconds.";
        } else {
            timeout_ = (uint32_t)int_timeout;
        }
    }

    void parse_ninfer(AttrList &model_config, NeuronDevice *neuron_device) {
        int64 max_num_threads = DEFAULT_MAX_NUM_INFER;
        if (model_config_valid(model_config)) {
            int64 opt_num_infer = model_config_this_opt_num_cores(model_config);
            if (opt_num_infer > 0 && opt_num_infer <= HARD_MAX_NUM_THREADS) {
                // add some extras for CPU nodes
                max_num_threads = opt_num_infer + NRTD_NUM_CPU_THREADS;
            } else {
                LOG(WARNING) << "model_config with opt_num_infer=" << opt_num_infer
                             << " is invalid; using default value "
                             << max_num_threads  << " instead.";
            }
        }
        std::string ninfer_str = env_get("NEURON_MAX_NUM_INFERS", "");
        bool num_infer_is_negative = false;
        if (!ninfer_str.empty()) {
            int64 env_ninfer = (int64)stoi_no_throw(ninfer_str);
            if (env_ninfer < -HARD_MAX_NUM_THREADS || env_ninfer > HARD_MAX_NUM_THREADS) {
                LOG(WARNING) << "NEURON_MAX_NUM_INFERS=" << ninfer_str
                             << " is invalid; using default value "
                             << max_num_threads  << " instead.";
            } else if (env_ninfer < 0) {
                num_infer_is_negative = true;
                max_num_threads = -env_ninfer;
            } else if (0 == env_ninfer) {
                LOG(WARNING) << "NEURON_MAX_NUM_INFERS=0 is invalid; using 1 instead.";
                max_num_threads = 1;
            } else {
                max_num_threads = env_ninfer;
            }
        }
        if (model_config_valid(model_config)) {
            // enforce max_num_threads = 1 if ncg size is insufficient
            int64 int64_opt_num_cores = model_config_this_opt_num_cores(model_config);
            if (int64_opt_num_cores < NeuronDeviceManager::MIN_NUM_CORES
                    || int64_opt_num_cores > NeuronDeviceManager::MAX_NUM_CORES) {
                max_num_threads = NRTD_INSUFFICIENT_NUM_INFER;
            } else {
                uint32_t opt_num_cores = (uint32_t)int64_opt_num_cores;
                if (neuron_device->num_cores() < opt_num_cores) {
                    max_num_threads = NRTD_INSUFFICIENT_NUM_INFER;
                }
            }
        }
        max_num_threads = max_num_threads > 1 ? max_num_threads : 1;
        max_num_threads = std::min(max_num_threads, HARD_MAX_NUM_THREADS);
        max_num_infers_ = (uint32_t)max_num_threads;
        ninfer_ = num_infer_is_negative ? max_num_infers_ : max_num_infers_ + 1;
    }

    void parse_device_index(AttrList &model_config) {
        device_index_ = model_config_device_index(model_config);
    }

    int64_t opt_device_size_ = -1;
    int64_t max_num_duplicates_ = 1;
    uint32_t max_num_infers_ = 4;
    uint32_t timeout_ = 2;
    uint32_t ninfer_ = 5;
    int64_t device_index_ = -1;
private:
    bool model_config_valid(AttrList &model_config) {
        return model_config.i_size() >= 4;
    }
    int64 model_config_global_opt_num_cores(AttrList &model_config) {
        return model_config.i(0);
    }
    int64 model_config_this_opt_num_cores(AttrList &model_config) {
        return model_config.i(1);
    }
    int64 model_config_max_num_duplicates(AttrList &model_config) {
        return model_config.i(2);
    }
    int64 model_config_timeout(AttrList &model_config) {
        return model_config.i(3);
    }
    int64 model_config_device_index(AttrList &model_config) {
        if (model_config.i_size() >= 5) {
            return model_config.i(4);
        } else {
            return -1;
        }
    }

    static const int64 DEFAULT_MAX_NUM_INFER = 4;
    static const int64 NRTD_INSUFFICIENT_NUM_INFER = 1;
    static const int64 NRTD_NUM_CPU_THREADS = 3;
    static const int64 HARD_MAX_NUM_THREADS = 1024;
};


NeuronOp::NeuronOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    VLOG(1) << "calling NeuronOp constructor";
    OP_REQUIRES(ctx, 0 != def().attr().at("executable").s().size(),
                errors::InvalidArgument(
                    "Neuron executable (neff) is empty. Please ensure you compiled ",
                    "the SavedModel with `tensorflow.neuron.saved_model.compile`.",
                    "If you specified argument `compiler_recovery=False`, this ",
                    "error message indicates that neuron-cc could not optimize ",
                    "you model for running on Neuron runtime."));
    profile_.initialize(env_get("NEURON_PROFILE"), def().name());
    if (profile_.enabled_) profile_.dump_info(def().attr().at("graph_def").s(),
                                              def().attr().at("executable").s());
    VLOG(1) << "NeuronOp constructor done";
}

static size_t get_tensor_size(const DataType dype, const TensorShapeProto &shape_proto) {
    size_t dtype_size = (size_t)DataTypeSize(dype);
    size_t num_elements = (size_t)TensorShape(shape_proto).num_elements();
    return dtype_size * num_elements;
}

static Status get_io_tensor_sizes(std::vector<size_t> *input_tensor_sizes,
                                  std::vector<size_t> *output_tensor_sizes,
                                  const NodeDef &node_def) {
    const google::protobuf::Map<std::string, AttrValue> &attr = node_def.attr();
    AttrList &input_names = attr.at("input_names").list();
    AttrList &input_dtypes = attr.at("input_dtypes").list();
    AttrList &input_shapes = attr.at("input_shapes").list();
    AttrList &output_names = attr.at("output_names").list();
    AttrList &output_dtypes = attr.at("output_dtypes").list();
    AttrList &output_shapes = attr.at("output_shapes").list();
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
    if (input_tensor_sizes != nullptr) {
        input_tensor_sizes->clear();
        for (auto idx = 0; idx < input_dtypes.type_size(); ++idx) {
            size_t tensor_size = get_tensor_size(input_dtypes.type(idx), input_shapes.shape(idx));
            input_tensor_sizes->push_back(tensor_size);
        }
    }
    if (output_tensor_sizes != nullptr) {
        output_tensor_sizes->clear();
        for (auto idx = 0; idx < output_dtypes.type_size(); ++idx) {
            size_t tensor_size = get_tensor_size(output_dtypes.type(idx), output_shapes.shape(idx));
            output_tensor_sizes->push_back(tensor_size);
        }
    }
    return Status::OK();
}

static Status check_input_tensors(const std::vector<const Tensor*> &input_tensors,
                                  const NodeDef &node_def) {
    AttrList &input_names = node_def.attr().at("input_names").list();
    std::vector<size_t> input_tensor_sizes;
    TF_RETURN_IF_ERROR(get_io_tensor_sizes(&input_tensor_sizes, nullptr, node_def));
    if ((int)input_tensors.size() != input_names.s_size()) {
        return errors::Internal(
            "incorrect number of input tensors, input_tensors size ",
            input_tensors.size(), ", input_names size", input_names.s_size());
    }
    for (auto idx = 0; idx < input_names.s_size(); ++idx) {
        size_t tensor_data_size = input_tensors[idx]->tensor_data().size();
        if (tensor_data_size != input_tensor_sizes[idx]) {
            return errors::Internal(
                "incorrect input tensor size ", tensor_data_size, " found on ",
                input_names.s(idx), " (", input_tensor_sizes[idx], ")");
        }
    }
    return Status::OK();
}

Status NeuronOp::initialize(const std::string &session_handle) {
    tensorflow::mutex_lock lock(mutex_model_);
    if (nullptr != neuron_device_) {
        VLOG(1) << "NeuronOp is already initialized";
        return Status::OK();
    }
    AttrList &model_config_attr = def().attr().at("model_config").list();
    NeuronModelConfig model_config;
    model_config.parse_opt_device_size(model_config_attr);
    model_config.parse_device_index(model_config_attr);
    TF_RETURN_IF_ERROR(
        global_neuron_device_manager.apply_for_device(
            &neuron_device_, session_handle,
            model_config.opt_device_size_, model_config.max_num_duplicates_,
            model_config.device_index_)
    );
    model_config.parse_timeout(model_config_attr);
    model_config.parse_ninfer(model_config_attr, neuron_device_);
    StringPiece executable(def().attr().at("executable").s());
    TF_RETURN_IF_ERROR(neuron_device_->load(&nn_id_, executable, model_config.timeout_,
                                            model_config.ninfer_, profile_.enabled_));
    VLOG(1) << "loaded " << def().name() << " as " << nn_id_
            << "; number of NEFFs: " << neuron_device_->num_executable();

    // check argument sizes
    TF_RETURN_IF_ERROR(get_io_tensor_sizes(nullptr, nullptr, def()));

    max_num_infers_ = model_config.max_num_infers_;
    max_num_infers_ *= neuron_device_->semaphore_factor();
    std::string unlimited_threads = env_get("NEURON_UNLIMITED_THREADS", "");
    if (!infer_sem_ && "yes" != unlimited_threads) {
        infer_sem_ = std::make_shared<xla::Semaphore>(max_num_infers_);
        VLOG(1) << "infer semaphore capacity " << max_num_infers_;
    }
    return Status::OK();
}

NeuronOp::~NeuronOp() {
    VLOG(1) << "calling NeuronOp destructor";
    tensorflow::mutex_lock lock(mutex_model_);
    if (nullptr == neuron_device_) {
        VLOG(1) << "neuron_device_ not available; not tearing down";
        return;
    }
    neuron_device_->unload(nn_id_);
    VLOG(1) << "unload from NeuronOp::~NeuronOp";
    global_neuron_device_manager.clear_if_empty();
    VLOG(1) << "NeuronOp destructor done";
}

void NeuronOp::Compute(OpKernelContext *ctx) {
    thread::ThreadPool *thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    Timestamps timestamps;
    timestamps.mark_enter();

    AttrList &input_names = def().attr().at("input_names").list();
    AttrList &output_shapes = def().attr().at("output_shapes").list();
    std::vector<const Tensor*> input_tensors(ctx->num_inputs());
    for (auto idx = 0; idx < ctx->num_inputs(); ++idx) {
        input_tensors[idx] = &ctx->input(idx);
    }
    OP_REQUIRES(ctx, (int)input_tensors.size() == input_names.s_size(),
                errors::InvalidArgument("incorrect number of input tensors"));
    std::vector<size_t> input_tensor_sizes;
    std::vector<size_t> output_tensor_sizes;
    OP_REQUIRES_OK(ctx, get_io_tensor_sizes(&input_tensor_sizes, &output_tensor_sizes, def()));

    int64_t batch_size = UNINIT_BATCH_SIZE;
    int64_t k_batch_size = UNINIT_BATCH_SIZE;
    std::vector<bool> is_batch_input_tensors(input_tensors.size());
    std::vector<bool> is_batch_output_tensors(ctx->num_outputs());
    bool use_dynamic_batch_size = false;
    AttrList &output_names = def().attr().at("output_names").list();
    AttrList &output_dtypes = def().attr().at("output_dtypes").list();
    AttrList &input_batch_axis = def().attr().at("input_batch_axis").list();
    AttrList &output_batch_axis = def().attr().at("output_batch_axis").list();
    bool enable_dynamic_batch_size = false;
    for (auto idx = 0; idx < input_batch_axis.i_size(); ++idx) {
        if (-1 != input_batch_axis.i(idx)) {
            enable_dynamic_batch_size = true;
            break;
        }
    }
    if (enable_dynamic_batch_size && input_names.s_size() == input_batch_axis.i_size() &&
            output_names.s_size() == output_batch_axis.i_size()) {
        AttrList &input_shapes = def().attr().at("input_shapes").list();
        for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
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
            is_batch_input_tensors[idx] = is_batch_tensor;
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
            is_batch_output_tensors[idx] = is_batch_tensor;
        }
    }
    OP_REQUIRES(ctx, ctx->num_outputs() == output_names.s_size(),
                errors::InvalidArgument("incorrect number of output tensors"));

    // keep a shared pointer so that RuntimeSession outlives shared memory buffers
    std::shared_ptr<RuntimeSession> session_alive;

    if (use_dynamic_batch_size) {
        int64_t pad_batch_size = ((batch_size - 1) / k_batch_size + 1) * k_batch_size;
        std::vector<Tensor*> batch_output_tensors(ctx->num_outputs());
        for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
            Tensor *batch_out_tensor = nullptr;
            TensorShape shape(output_shapes.shape(idx));
            if (is_batch_output_tensors[idx]) {
                shape.set_dim(0, batch_size);
            }
            OP_REQUIRES_OK(ctx, ctx->allocate_output(idx, shape, &batch_out_tensor));
            batch_output_tensors[idx] = batch_out_tensor;
        }

        int64_t num_batches = pad_batch_size / k_batch_size;
        std::vector<std::vector<Tensor> > batches_neuron_input_tensors(num_batches);
        for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int64_t dim0_start = batch_idx * k_batch_size;
            int64_t dim0_limit = batch_idx * k_batch_size + k_batch_size;
            for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
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
                            thread_pool, &pad_end_slice, t_data, t_data.size()));
                        batches_neuron_input_tensors[batch_idx].emplace_back(pad_end_slice);
                    } else {
                        batches_neuron_input_tensors[batch_idx].emplace_back(
                            input_tensors[idx]->Slice(dim0_start, dim0_limit));
                    }
                } else {
                    batches_neuron_input_tensors[batch_idx].emplace_back();
                }
            }
        }

        OK_IGNORE_ABORTED(ctx, initialize(ctx->session_handle()));
        session_alive = neuron_device_->get_session();

        int64_t window_size = max_num_infers_ > 1 ? max_num_infers_ : 1;
        window_size = std::min(window_size, num_batches);

        // run an extra inference upfront if profiler is enabled
        if (profile_.enabled_) {
            std::vector<const Tensor*> sliced_inputs(input_names.s_size());
            for (auto idx = 0; idx < input_names.s_size(); ++idx) {
                sliced_inputs[idx] = is_batch_input_tensors[idx] ?
                    &batches_neuron_input_tensors[0][idx] : input_tensors[idx];
            }
            OP_REQUIRES_OK(ctx, check_input_tensors(sliced_inputs, def()));
            std::vector<Tensor> temp_outputs(output_dtypes.type_size());
            for (auto idx = 0; idx < output_dtypes.type_size(); ++idx) {
                OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                    output_dtypes.type(idx), output_shapes.shape(idx),
                    &temp_outputs[idx]));
            }
            std::vector<Tensor*> output_tensors(temp_outputs.size());
            for (size_t idx = 0; idx < temp_outputs.size(); ++idx) {
                output_tensors[idx] = &temp_outputs[idx];
            }
            ScopedRuntimeIO scoped_io;
            OK_IGNORE_ABORTED(ctx, scoped_io.setup(
                input_names, input_tensor_sizes, sliced_inputs,
                output_names, output_tensor_sizes, output_tensors,
                nn_id_, thread_pool, neuron_device_->shm_buf_mgr_));
            OP_REQUIRES_OK(ctx, neuron_device_->infer(
                &scoped_io.runtime_io_, nullptr, &profile_, nn_id_));
            OK_IGNORE_ABORTED(ctx, scoped_io.finish());
        }

        std::queue<ScopedRuntimeIO> scoped_io_queue;
        std::vector<std::vector<Tensor> > batches_sliced_outputs(num_batches);
        {   // scope of semaphore reservation queue
            SemResQueue sem_res_queue;
            {   // lock device
                std::queue<tensorflow::mutex_lock> mutex_lock_queue;
                std::queue<RuntimeIO*> need_wait_infer_post;
                int64_t first_need_wait_infer_post_bidx = num_batches - window_size;
                neuron_device_->acquire_mutex(&mutex_lock_queue);
                OK_IGNORE_ABORTED(ctx, neuron_device_->start_model_unsafe(nn_id_));
                // need an extra unary grpc call to re-establish channel in case of seeing grpc 14
                // as start_model_unsafe may not call grpc start
                OK_IGNORE_ABORTED(ctx, neuron_device_->start_ping(nn_id_));
                // post ninfer ones
                for (int64_t post_bidx = 0; post_bidx < window_size; ++post_bidx) {
                    // setup inputs
                    std::vector<const Tensor*> sliced_inputs(input_names.s_size());
                    for (auto idx = 0; idx < input_names.s_size(); ++idx) {
                        sliced_inputs[idx] = is_batch_input_tensors[idx] ?
                            &batches_neuron_input_tensors[post_bidx][idx] : input_tensors[idx];
                    }
                    OP_REQUIRES_OK(ctx, check_input_tensors(sliced_inputs, def()));

                    // setup outputs
                    int64_t dim0_start = post_bidx * k_batch_size;
                    int64_t dim0_limit = post_bidx * k_batch_size + k_batch_size;
                    for (auto idx = 0; idx < output_dtypes.type_size(); ++idx) {
                        if (is_batch_output_tensors[idx]) {
                            Tensor slice = batch_output_tensors[idx]->Slice(
                                dim0_start, std::min(dim0_limit, batch_size));
                            batches_sliced_outputs[post_bidx].push_back(slice);
                        } else {
                            batches_sliced_outputs[post_bidx].emplace_back();
                        }
                    }
                    std::vector<Tensor*> output_tensors(output_dtypes.type_size());
                    for (auto idx = 0; idx < output_dtypes.type_size(); ++idx) {
                        output_tensors[idx] = is_batch_output_tensors[idx] ?
                            &batches_sliced_outputs[post_bidx][idx] : batch_output_tensors[idx];
                    }
                    scoped_io_queue.emplace();
                    OK_IGNORE_ABORTED(ctx, scoped_io_queue.back().setup(
                        input_names, input_tensor_sizes, sliced_inputs,
                        output_names, output_tensor_sizes, output_tensors,
                        nn_id_, thread_pool, neuron_device_->shm_buf_mgr_));
                    if (infer_sem_) {
                        sem_res_queue.push(infer_sem_->ScopedAcquire(1));
                    }

                    // post
                    RuntimeIO *runtime_io = &scoped_io_queue.back().runtime_io_;
                    if (post_bidx >= first_need_wait_infer_post_bidx) {
                        OP_REQUIRES_OK(ctx, neuron_device_->setup_infer_post(runtime_io, post_bidx));
                        if (0 == post_bidx) {
                            timestamps.mark_above_nrtd_infer();
                        }
                        OP_REQUIRES_OK(ctx, neuron_device_->post_infer_post(runtime_io));
                        need_wait_infer_post.push(runtime_io);
                    } else {
                        OP_REQUIRES_OK(ctx, neuron_device_->setup_infer(runtime_io, post_bidx));
                        if (0 == post_bidx) {
                            timestamps.mark_above_nrtd_infer();
                        }
                        OP_REQUIRES_OK(ctx, neuron_device_->post_infer(runtime_io));
                    }
                }

                // wait one and post one
                for (int64_t post_bidx = window_size; post_bidx < num_batches; ++post_bidx) {
                    // setup inputs for next one
                    std::vector<const Tensor*> sliced_inputs(input_names.s_size());
                    for (auto idx = 0; idx < input_names.s_size(); ++idx) {
                        sliced_inputs[idx] = is_batch_input_tensors[idx] ?
                            &batches_neuron_input_tensors[post_bidx][idx] : input_tensors[idx];
                    }
                    OP_REQUIRES_OK(ctx, check_input_tensors(sliced_inputs, def()));

                    // setup outputs for next one
                    int64_t dim0_start = post_bidx * k_batch_size;
                    int64_t dim0_limit = post_bidx * k_batch_size + k_batch_size;
                    for (auto idx = 0; idx < output_dtypes.type_size(); ++idx) {
                        if (is_batch_output_tensors[idx]) {
                            Tensor slice = batch_output_tensors[idx]->Slice(
                                dim0_start, std::min(dim0_limit, batch_size));
                            batches_sliced_outputs[post_bidx].push_back(slice);
                        } else {
                            batches_sliced_outputs[post_bidx].emplace_back();
                        }
                    }
                    std::vector<Tensor*> output_tensors(output_dtypes.type_size());
                    for (auto idx = 0; idx < output_dtypes.type_size(); ++idx) {
                        output_tensors[idx] = is_batch_output_tensors[idx] ?
                            &batches_sliced_outputs[post_bidx][idx] : batch_output_tensors[idx];
                    }
                    scoped_io_queue.emplace();
                    OK_IGNORE_ABORTED(ctx, scoped_io_queue.back().setup(
                        input_names, input_tensor_sizes, sliced_inputs,
                        output_names, output_tensor_sizes, output_tensors,
                        nn_id_, thread_pool, neuron_device_->shm_buf_mgr_));
                    RuntimeIO *runtime_io_back = &scoped_io_queue.back().runtime_io_;
                    if (post_bidx >= first_need_wait_infer_post_bidx) {
                        OP_REQUIRES_OK(ctx, neuron_device_->setup_infer_post(runtime_io_back, post_bidx));
                    } else {
                        OP_REQUIRES_OK(ctx, neuron_device_->setup_infer(runtime_io_back, post_bidx));
                    }

                    // wait one
                    RuntimeIO *runtime_io_front = &scoped_io_queue.front().runtime_io_;
                    OP_REQUIRES_OK(ctx, neuron_device_->wait_infer(runtime_io_front));

                    // post next one
                    if (post_bidx >= first_need_wait_infer_post_bidx) {
                        OP_REQUIRES_OK(ctx, neuron_device_->post_infer_post(runtime_io_back));
                        need_wait_infer_post.push(runtime_io_back);
                    } else {
                        OP_REQUIRES_OK(ctx, neuron_device_->post_infer(runtime_io_back));
                    }
                    OK_IGNORE_ABORTED(ctx, runtime_io_front->finish());
                    scoped_io_queue.pop();
                }

                // ensure all infer_post calls are queued up by RT before releasing eg lock
                OP_REQUIRES(ctx, (int64_t)need_wait_infer_post.size() == window_size,
                            errors::Internal("incorrect queue length found -- race condition likely"));
                for (int64_t wait_bidx = 0; wait_bidx < window_size; ++wait_bidx) {
                    RuntimeIO *runtime_io_front = need_wait_infer_post.front();
                    OP_REQUIRES_OK(ctx, neuron_device_->wait_infer_post(runtime_io_front));
                    need_wait_infer_post.pop();
                }
            }   // unlock device

            // wait for remaining ones in the queue
            for (int64_t wait_bidx = 0; wait_bidx < window_size; ++wait_bidx) {
                if (scoped_io_queue.empty()) {
                    break;
                }
                Timestamps *wait_timestamps = (window_size - 1) == wait_bidx ?
                    &timestamps : nullptr;
                RuntimeIO *runtime_io_front = &scoped_io_queue.front().runtime_io_;
                OP_REQUIRES_OK(ctx, neuron_device_->infer_wait(runtime_io_front, wait_timestamps));
                sem_res_queue.pop();
                OK_IGNORE_ABORTED(ctx, runtime_io_front->finish());
                scoped_io_queue.pop();
            }
        }   // semaphore reservation queue goes out of scope
    } else if (profile_.enabled_) {
        VLOG(1) << "profile enabled -- lock stop/start/infer altogether";
        std::vector<Tensor*> output_tensors(ctx->num_outputs());
        for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(idx, output_shapes.shape(idx),
                                                     &output_tensors[idx]));
        }
        OK_IGNORE_ABORTED(ctx, initialize(ctx->session_handle()));
        session_alive = neuron_device_->get_session();
        OP_REQUIRES_OK(ctx, check_input_tensors(input_tensors, def()));
        ScopedRuntimeIO scoped_io;
        OK_IGNORE_ABORTED(ctx, scoped_io.setup(
            input_names, input_tensor_sizes, input_tensors,
            output_names, output_tensor_sizes, output_tensors,
            nn_id_, thread_pool, neuron_device_->shm_buf_mgr_));
        OK_IGNORE_ABORTED(ctx, neuron_device_->infer(
            &scoped_io.runtime_io_, &timestamps, &profile_, nn_id_));
        OP_REQUIRES_OK(ctx, scoped_io.finish());
    } else {
        std::vector<Tensor*> output_tensors(ctx->num_outputs());
        for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(idx, output_shapes.shape(idx),
                                                     &output_tensors[idx]));
        }
        OK_IGNORE_ABORTED(ctx, initialize(ctx->session_handle()));
        session_alive = neuron_device_->get_session();
        OP_REQUIRES_OK(ctx, check_input_tensors(input_tensors, def()));
        ScopedRuntimeIO scoped_io;
        OK_IGNORE_ABORTED(ctx, scoped_io.setup(
            input_names, input_tensor_sizes, input_tensors,
            output_names, output_tensor_sizes, output_tensors,
            nn_id_, thread_pool, neuron_device_->shm_buf_mgr_));
        {
            SemResQueue sem_res_queue;
            OK_IGNORE_ABORTED(ctx, neuron_device_->infer_post(
                &scoped_io.runtime_io_, &sem_res_queue, infer_sem_, &timestamps, nn_id_));
            OP_REQUIRES_OK(ctx, neuron_device_->infer_wait(&scoped_io.runtime_io_, &timestamps));
        }
        OK_IGNORE_ABORTED(ctx, scoped_io.finish());
    }
    timestamps.mark_exit();
    VLOG(1) << timestamps.timing_string();
}

// need to override kernel_builder as NeuronName to prevent multiple kernel registrations
#if TF_VERSION_LESS_THAN(2, 4)
REGISTER_KERNEL_BUILDER(NeuronName(), neuron::NeuronOp);
#else
REGISTER_KERNEL_BUILDER_IMPL_2("NeuronOp", register_kernel::NeuronName(), false, NeuronOp);
#endif

}  // namespace neuron
}  // namespace tensorflow

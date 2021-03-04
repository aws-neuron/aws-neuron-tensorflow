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
#include "model_config.h"
#include "model.h"


#define TFNN_ASSERT(cond, error) {      \
    if (TF_PREDICT_FALSE(!(cond))) {    \
        return (error);                 \
    }                                   \
}

// Note: this macro cannot appear before ctx->allocate_output
#define RIE_IGNORE_ABORTED(...) {                                               \
    Status status(__VA_ARGS__);                                                 \
    if (TF_PREDICT_TRUE(status.code() != tensorflow::error::Code::ABORTED)) {   \
        TF_RETURN_IF_ERROR(status);                                             \
    }                                                                           \
}


namespace tensorflow {
namespace neuron {


static const int64 UNINIT_BATCH_SIZE = -8;  // magic number for uninitialized batch size

static size_t get_tensor_size(const DataType dype, const TensorShapeProto &shape_proto) {
    size_t dtype_size = (size_t)DataTypeSize(dype);
    size_t num_elements = (size_t)TensorShape(shape_proto).num_elements();
    return dtype_size * num_elements;
}

static Status get_io_tensor_sizes(std::vector<size_t> *tensor_sizes,
                                  const NodeDef &node_def,
                                  const std::string &io_type) {
    if (TF_PREDICT_FALSE(io_type != "input" && io_type != "output")) {
        return errors::InvalidArgument("io_type must be one of {input, output}; got ", io_type);
    }
    const google::protobuf::Map<std::string, AttrValue> &attr = node_def.attr();
    AttrList &names = attr.at(io_type + "_names").list();
    AttrList &dtypes = attr.at(io_type + "_dtypes").list();
    AttrList &shapes = attr.at(io_type + "_shapes").list();
    if (TF_PREDICT_FALSE(
            names.s_size() != dtypes.type_size() || names.s_size() != shapes.shape_size())) {
        return errors::FailedPrecondition(
            "incorrect number of tensors: ", io_type, "_names size ", names.s_size(),
            ", ", io_type, "_dtypes size ", dtypes.type_size(),
            ", ", io_type, "_shapes size ", shapes.shape_size());
    }
    if (TF_PREDICT_TRUE(tensor_sizes != nullptr)) {
        tensor_sizes->clear();
        for (auto idx = 0; idx < dtypes.type_size(); ++idx) {
            size_t tensor_size = get_tensor_size(dtypes.type(idx), shapes.shape(idx));
            tensor_sizes->push_back(tensor_size);
        }
    }
    return Status::OK();
}

static Status check_input_tensors(const std::vector<const Tensor*> &input_tensors,
                                  const NodeDef &node_def) {
    AttrList &input_names = node_def.attr().at("input_names").list();
    AttrList &input_dtypes = node_def.attr().at("input_dtypes").list();
    AttrList &input_shapes = node_def.attr().at("input_shapes").list();
    TFNN_ASSERT((int)input_tensors.size() == input_names.s_size(),
                errors::Internal(
                    "incorrect number of input tensors, input_tensors size ",
                    input_tensors.size(), ", input_names size", input_names.s_size()));
    TFNN_ASSERT(input_dtypes.type_size() == input_names.s_size(),
                errors::Internal(
                    "incorrect input metadata, input_dtypes size ",
                    input_dtypes.type_size(), ", input_names size", input_names.s_size()));
    TFNN_ASSERT(input_shapes.shape_size() == input_names.s_size(),
                errors::Internal(
                    "incorrect input metadata, input_shapes size ",
                    input_shapes.shape_size(), ", input_names size", input_names.s_size()));
    for (auto idx = 0; idx < input_dtypes.type_size(); ++idx) {
        DataType dtype = input_tensors[idx]->dtype();
        TensorShape shape = input_tensors[idx]->shape();
        DataType dtype_expected = input_dtypes.type(idx);
        TensorShape shape_expected = input_shapes.shape(idx);
        TFNN_ASSERT(dtype == dtype_expected,
                    errors::Internal(
                        "incorrect input tensor dtype ", dtype, ", expected ", dtype_expected));
        TFNN_ASSERT(shape == shape_expected,
                    errors::Internal(
                        "incorrect input tensor shape ", shape, ", expected ", shape_expected));
    }
    return Status::OK();
}


Status NeuronModel::initialize(const NodeDef &node_def, const std::string &session_handle) {
    tensorflow::mutex_lock lock(mutex_model_);
    if (TF_PREDICT_TRUE(nullptr != neuron_device_)) {
        VLOG(1) << "NeuronModel is already initialized";
        return Status::OK();
    }
    const google::protobuf::Map<std::string, AttrValue> &attr = node_def.attr();
    if (0 == attr.at("executable").s().size()) {
        return errors::InvalidArgument("Neuron executable (neff) is empty.");
    }
    profile_.initialize(env_get("NEURON_PROFILE"), node_def.name());
    if (profile_.enabled_) profile_.dump_info(attr.at("graph_def").s(),
                                              attr.at("executable").s());
    AttrList &model_config_attr = attr.at("model_config").list();
    NeuronModelConfig model_config;
    model_config.parse_opt_device_size(model_config_attr);
    model_config.parse_device_index(model_config_attr);
    TF_RETURN_IF_ERROR(
        NeuronDeviceManager::GetNeuronDeviceManager().apply_for_device(
            &neuron_device_, session_handle,
            model_config.opt_device_size_, model_config.max_num_duplicates_,
            model_config.device_index_)
    );
    model_config.parse_timeout(model_config_attr);
    model_config.parse_ninfer(
        model_config_attr, neuron_device_->num_cores(),
        NeuronDeviceManager::MIN_NUM_CORES, NeuronDeviceManager::MAX_NUM_CORES);
    StringPiece executable(attr.at("executable").s());
    TF_RETURN_IF_ERROR(neuron_device_->load(&nn_id_, executable, model_config.timeout_,
                                            model_config.ninfer_, profile_.enabled_));
    VLOG(1) << "loaded " << node_def.name() << " as " << nn_id_
            << "; number of NEFFs: " << neuron_device_->num_executable();

    // check argument sizes
    TF_RETURN_IF_ERROR(get_io_tensor_sizes(nullptr, node_def, "input"));
    TF_RETURN_IF_ERROR(get_io_tensor_sizes(nullptr, node_def, "output"));

    max_num_infers_ = model_config.max_num_infers_;
    max_num_infers_ *= neuron_device_->semaphore_factor();
    std::string unlimited_threads = env_get("NEURON_UNLIMITED_THREADS", "");
    if (!infer_sem_ && "yes" != unlimited_threads) {
        infer_sem_ = std::make_shared<xla::Semaphore>(max_num_infers_);
        VLOG(1) << "infer semaphore capacity " << max_num_infers_;
    }
    return Status::OK();
}

static Status allocate_shuffle_buffers(OpKernelContext *ctx,
                                       std::vector<Tensor> *shuffle_buffers,
                                       const std::vector<const Tensor*> &input_tensors) {
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
        const Tensor *src = input_tensors[idx];
        Tensor *dst = &shuffle_buffers->at(idx);
        TF_RETURN_IF_ERROR(ctx->allocate_temp(src->dtype(), src->shape(), dst));
    }
    return Status::OK();
}

static Status copy_input_tensors_with_shuffle(OpKernelContext *ctx, const NodeDef &node_def,
                                              const std::vector<const Tensor*> &input_tensors,
                                              ScopedRuntimeIO *scoped_io) {
    const google::protobuf::Map<std::string, AttrValue> &attr = node_def.attr();
    if (attr.count("_input_shuffles")) {
        AttrList &input_shuffles = attr.at("_input_shuffles").list();
        if (TF_PREDICT_FALSE(input_shuffles.tensor_size() != (int64)input_tensors.size())) {
            return errors::InvalidArgument("illegal _input_shuffles attribute");
        }
        std::vector<Tensor> shuffle_buffers(input_tensors.size());
        TF_RETURN_IF_ERROR(allocate_shuffle_buffers(ctx, &shuffle_buffers, input_tensors));
        RIE_IGNORE_ABORTED(scoped_io->copy_input_tensors(
            input_tensors, input_shuffles, &shuffle_buffers));
    } else {
        RIE_IGNORE_ABORTED(scoped_io->copy_input_tensors(input_tensors));
    }
    return Status::OK();
}

Status NeuronModel::compute(OpKernelContext *ctx, const NodeDef &node_def,
                            const std::vector<const Tensor*> &input_tensors) {
    thread::ThreadPool *thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    Timestamps timestamps;
    timestamps.mark_enter();

    const google::protobuf::Map<std::string, AttrValue> &attr = node_def.attr();
    AttrList &input_names = attr.at("input_names").list();
    AttrList &output_shapes = attr.at("output_shapes").list();
    TFNN_ASSERT((int)input_tensors.size() == input_names.s_size(),
                errors::InvalidArgument("incorrect number of input tensors"));
    std::vector<size_t> output_tensor_sizes;
    TF_RETURN_IF_ERROR(get_io_tensor_sizes(&output_tensor_sizes, node_def, "output"));

    int64_t batch_size = UNINIT_BATCH_SIZE;
    int64_t k_batch_size = UNINIT_BATCH_SIZE;
    std::vector<bool> is_batch_inputs(input_tensors.size());
    std::vector<bool> is_batch_outputs(ctx->num_outputs());
    bool use_dynamic_batch_size = false;
    AttrList &output_names = attr.at("output_names").list();
    AttrList &output_dtypes = attr.at("output_dtypes").list();
    AttrList &input_batch_axis = attr.at("input_batch_axis").list();
    AttrList &output_batch_axis = attr.at("output_batch_axis").list();
    bool enable_dynamic_batch_size = false;
    for (auto idx = 0; idx < input_batch_axis.i_size(); ++idx) {
        if (-1 != input_batch_axis.i(idx)) {
            enable_dynamic_batch_size = true;
            break;
        }
    }
    if (enable_dynamic_batch_size && input_names.s_size() == input_batch_axis.i_size() &&
            output_names.s_size() == output_batch_axis.i_size()) {
        AttrList &input_shapes = attr.at("input_shapes").list();
        for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
            bool is_batch_tensor = false;
            const Tensor *tptr = input_tensors[idx];
            TensorShape shape(tptr->shape());
            TensorShape k_shape(input_shapes.shape(idx));
            if (0 == input_batch_axis.i(idx)) {
                TFNN_ASSERT(shape.dims() > 0,
                            errors::InvalidArgument(
                                "no batch-dimension found on input tensor ",
                                input_names.s(idx), " with shape ", shape.DebugString()));
                if (UNINIT_BATCH_SIZE == batch_size) {
                    batch_size = shape.dim_size(0);
                    k_batch_size = k_shape.dim_size(0);
                    TFNN_ASSERT(batch_size > 0,
                                errors::Internal(
                                    "incorrect internal batch size inferred from input tensor ",
                                    input_names.s(idx), " with shape ", shape.DebugString()));
                } else {
                    TFNN_ASSERT(batch_size == shape.dim_size(0),
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
            TFNN_ASSERT(shape == k_shape,
                        errors::InvalidArgument(
                            "incorrect shape found on input tensor ", input_names.s(idx),
                            ", inference time shape ", tptr->shape().DebugString(),
                            ", expected shape ", input_shapes.shape(idx).DebugString()));
            is_batch_inputs[idx] = is_batch_tensor;
        }
        for (auto idx = 0; idx < output_names.s_size(); ++idx) {
            bool is_batch_tensor = false;
            if (0 == output_batch_axis.i(idx)) {
                TensorShape k_shape(output_shapes.shape(idx));
                TFNN_ASSERT(k_shape.dims() > 0,
                            errors::InvalidArgument(
                                "no batch-dimension found on output tensor ",
                                output_names.s(idx), " with Neuron shape ",
                                k_shape.DebugString()));
                TFNN_ASSERT(k_batch_size == k_shape.dim_size(0),
                            errors::InvalidArgument(
                                "incorrect batch size found on output tensor ",
                                output_names.s(idx), ", Neuron tensor shape ",
                                k_shape.DebugString(), ", Neuron batch size ",
                                k_batch_size));
                is_batch_tensor = batch_size != k_shape.dim_size(0);
            }
            is_batch_outputs[idx] = is_batch_tensor;
        }
    }
    TFNN_ASSERT(ctx->num_outputs() == output_names.s_size(),
                errors::InvalidArgument("incorrect number of output tensors"));

    // keep a shared pointer so that RuntimeSession outlives shared memory buffers
    std::shared_ptr<RuntimeSession> session_alive;

    if (use_dynamic_batch_size) {
        int64_t pad_batch_size = ((batch_size - 1) / k_batch_size + 1) * k_batch_size;
        VLOG(1) << "batch_size=" << batch_size;
        VLOG(1) << "k_batch_size=" << k_batch_size;
        VLOG(1) << "pad_batch_size=" << pad_batch_size;
        std::vector<Tensor*> output_tensors(ctx->num_outputs());
        for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
            Tensor *batch_out_tensor = nullptr;
            TensorShape shape(output_shapes.shape(idx));
            if (is_batch_outputs[idx]) {
                shape.set_dim(0, batch_size);
            }
            TF_RETURN_IF_ERROR(ctx->allocate_output(idx, shape, &batch_out_tensor));
            output_tensors[idx] = batch_out_tensor;
        }

        RIE_IGNORE_ABORTED(initialize(node_def, ctx->session_handle()));
        session_alive = neuron_device_->get_session();

        int64 end_start = k_batch_size - (pad_batch_size - batch_size);
        bool run_profiler_in_shard = false;
        Status shared_status;
        #define SHARD_LOG_RETURN_IF_ERROR(shared_status, ...) {                 \
            Status _status = (__VA_ARGS__);                                     \
            if (TF_PREDICT_FALSE(!_status.ok())) {                              \
                LOG(ERROR) << "shard error code " << _status.code()             \
                           << ", error message " << _status.error_message();    \
                shared_status = _status;                                        \
                return;                                                         \
            }                                                                   \
        }
        #define SHARD_LOG_IGNORE_ABORTED(shared_status, ...) {                              \
            Status _status(__VA_ARGS__);                                                    \
            if (TF_PREDICT_FALSE(!(_status.ok() ||                                          \
                                   _status.code() == tensorflow::error::Code::ABORTED))) {  \
                LOG(ERROR) << "shard error code " << _status.code()                         \
                           << ", error message " << _status.error_message();                \
                shared_status = _status;                                                    \
                return;                                                                     \
            }                                                                               \
        }
        auto ShardFunc = [
                this, &shared_status, &ctx, &node_def, &run_profiler_in_shard, &timestamps,
                &input_tensors, &is_batch_inputs, &input_names,
                &output_tensors, &is_batch_outputs, &output_names, &output_tensor_sizes,
                &batch_size, &k_batch_size, &end_start](int64 dim0_start, int64 dim0_limit) {
            if (TF_PREDICT_FALSE(dim0_limit - dim0_start != k_batch_size)) {
                shared_status = errors::Internal("illegal shard ", dim0_start, ":", dim0_limit);
                return;
            }
            VLOG(1) << "Sharding " << dim0_start << " to " << dim0_limit;
            std::vector<Tensor> sliced_inputs(input_tensors.size());
            for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
                if (is_batch_inputs[idx]) {
                    if (dim0_limit > batch_size) {
                        TensorShape ps_shape(input_tensors[idx]->shape());
                        ps_shape.set_dim(0, k_batch_size);
                        Tensor pad_end_slice(input_tensors[idx]->dtype(), ps_shape);
                        Tensor zero_slice = pad_end_slice.Slice(end_start, k_batch_size);
                        SHARD_LOG_RETURN_IF_ERROR(shared_status, tensor_memset(&zero_slice, 0));
                        Tensor end_slice = input_tensors[idx]->Slice(dim0_start, batch_size);
                        StringPiece t_data = end_slice.tensor_data();
                        SHARD_LOG_RETURN_IF_ERROR(shared_status, tensor_memcpy(
                            nullptr, &pad_end_slice, t_data, t_data.size()));
                        sliced_inputs[idx] = pad_end_slice;
                    } else {
                        sliced_inputs[idx] = input_tensors[idx]->Slice(dim0_start, dim0_limit);
                    }
                }
            }
            std::vector<const Tensor*> input_ptrs(input_tensors.size());
            for (auto i = 0; i < input_tensors.size(); ++i) {
                input_ptrs[i] = is_batch_inputs[i] ? &sliced_inputs[i] : input_tensors[i];
            }
            SHARD_LOG_RETURN_IF_ERROR(shared_status, check_input_tensors(input_ptrs, node_def));
            int64 end_limit = dim0_limit < batch_size ? dim0_limit : batch_size;
            std::vector<Tensor> sliced_outputs(output_tensors.size());
            for (auto i = 0; i < output_tensors.size(); ++i) {
                if (is_batch_outputs[i]) {
                    sliced_outputs[i] = output_tensors[i]->Slice(dim0_start, end_limit);
                }
            }
            std::vector<Tensor*> output_ptrs(output_tensors.size());
            for (auto i = 0; i < output_tensors.size(); ++i) {
                output_ptrs[i] = is_batch_outputs[i] ? &sliced_outputs[i] : output_tensors[i];
            }
            ScopedRuntimeIO scoped_io;
            SHARD_LOG_IGNORE_ABORTED(shared_status, neuron_device_->setup_scoped_runtime_io(
                &scoped_io, input_names, input_ptrs,
                output_names, output_tensor_sizes, output_ptrs, nn_id_, nullptr));

            // copy input tensors with optional input_shuffles
            SHARD_LOG_IGNORE_ABORTED(shared_status, copy_input_tensors_with_shuffle(
                ctx, node_def, input_ptrs, &scoped_io));

            // run inference
            if (TF_PREDICT_FALSE(run_profiler_in_shard)) {
                VLOG(1) << "enabling profiler in shard";
                SHARD_LOG_IGNORE_ABORTED(shared_status, neuron_device_->infer_with_profiling(
                    &scoped_io.runtime_io_, &timestamps, &profile_));
            } else {
                SHARD_LOG_IGNORE_ABORTED(shared_status, neuron_device_->infer(
                    &scoped_io.runtime_io_, infer_sem_, &timestamps));
            }
            SHARD_LOG_IGNORE_ABORTED(shared_status, scoped_io.finish());
        };
        #undef SHARD_LOG_IGNORE_ABORTED
        #undef SHARD_LOG_RETURN_IF_ERROR
        if (profile_.enabled_) {
            run_profiler_in_shard = true;
            ShardFunc(0, k_batch_size);
            run_profiler_in_shard = false;
            RIE_IGNORE_ABORTED(shared_status);
        }
        thread_pool->TransformRangeConcurrently(k_batch_size, pad_batch_size, std::move(ShardFunc));
        RIE_IGNORE_ABORTED(shared_status);
    } else {
        std::vector<Tensor*> output_tensors(ctx->num_outputs());
        for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
            TF_RETURN_IF_ERROR(
                ctx->allocate_output(idx, output_shapes.shape(idx), &output_tensors[idx]));
        }
        RIE_IGNORE_ABORTED(initialize(node_def, ctx->session_handle()));
        session_alive = neuron_device_->get_session();
        TF_RETURN_IF_ERROR(check_input_tensors(input_tensors, node_def));
        ScopedRuntimeIO scoped_io;
        RIE_IGNORE_ABORTED(neuron_device_->setup_scoped_runtime_io(
            &scoped_io, input_names, input_tensors,
            output_names, output_tensor_sizes, output_tensors, nn_id_, thread_pool));

        // copy input tensors with optional input_shuffles
        RIE_IGNORE_ABORTED(copy_input_tensors_with_shuffle(ctx, node_def, input_tensors, &scoped_io));

        // run inference
        if (profile_.enabled_) {
            VLOG(1) << "profile enabled -- lock stop/start/infer altogether";
            RIE_IGNORE_ABORTED(neuron_device_->infer_with_profiling(
                &scoped_io.runtime_io_, &timestamps, &profile_));
        } else {
            RIE_IGNORE_ABORTED(neuron_device_->infer(
                &scoped_io.runtime_io_, infer_sem_, &timestamps));
        }
        RIE_IGNORE_ABORTED(scoped_io.finish());
    }
    timestamps.mark_exit();
    VLOG(1) << timestamps.timing_string();
    return Status::OK();
}

NeuronModel::~NeuronModel() {
    VLOG(1) << "calling NeuronModel destructor";
    tensorflow::mutex_lock lock(mutex_model_);
    if (nullptr == neuron_device_) {
        VLOG(1) << "neuron_device_ not available; not tearing down";
        return;
    }
    neuron_device_->unload(nn_id_);
    VLOG(1) << "unload from NeuronModel::~NeuronModel";
    NeuronDeviceManager::GetNeuronDeviceManager().clear_if_empty();
    VLOG(1) << "NeuronModel destructor done";
}


}  // namespace neuron
}  // namespace tensorflow

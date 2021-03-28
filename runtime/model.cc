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

#include "model.h"
#include "device.h"
#include "engine.h"
#include "model_config.h"

#define TFNN_ASSERT(cond, error)     \
  {                                  \
    if (TF_PREDICT_FALSE(!(cond))) { \
      return (error);                \
    }                                \
  }

// Note: this macro cannot appear before ctx->allocate_output
#define RIE_IGNORE_ABORTED(...)                                               \
  {                                                                           \
    Status status(__VA_ARGS__);                                               \
    if (TF_PREDICT_TRUE(status.code() != tensorflow::error::Code::ABORTED)) { \
      TF_RETURN_IF_ERROR(status);                                             \
    }                                                                         \
  }

namespace tensorflow {
namespace neuron {

// some magic numbers
static const int64 UNINIT_BATCH_SIZE = -8;
static const int64 STATIC_BATCH_AXIS = -1;
static const int64 H2D_POOL_SIZE = 8;

static size_t get_tensor_size(const DataType dype,
                              const TensorShapeProto& shape_proto) {
  size_t dtype_size = (size_t)DataTypeSize(dype);
  size_t num_elements = (size_t)TensorShape(shape_proto).num_elements();
  return dtype_size * num_elements;
}

static Status get_io_tensor_sizes(std::vector<size_t>* tensor_sizes,
                                  const NodeDef& node_def,
                                  const std::string& io_type) {
  if (TF_PREDICT_FALSE(io_type != "input" && io_type != "output")) {
    return errors::InvalidArgument(
        "io_type must be one of {input, output}; got ", io_type);
  }
  const google::protobuf::Map<std::string, AttrValue>& attr = node_def.attr();
  AttrList& names = attr.at(io_type + "_names").list();
  AttrList& dtypes = attr.at(io_type + "_dtypes").list();
  AttrList& shapes = attr.at(io_type + "_shapes").list();
  if (TF_PREDICT_FALSE(names.s_size() != dtypes.type_size() ||
                       names.s_size() != shapes.shape_size())) {
    return errors::FailedPrecondition(
        "incorrect number of tensors: ", io_type, "_names size ",
        names.s_size(), ", ", io_type, "_dtypes size ", dtypes.type_size(),
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

static Status check_input_tensors(const std::vector<Tensor>& input_tensors,
                                  const NodeDef& node_def) {
  AttrList& input_names = node_def.attr().at("input_names").list();
  AttrList& input_dtypes = node_def.attr().at("input_dtypes").list();
  AttrList& input_shapes = node_def.attr().at("input_shapes").list();
  TFNN_ASSERT(
      (int64)input_tensors.size() == input_names.s_size(),
      errors::Internal("incorrect number of input tensors, input_tensors size ",
                       input_tensors.size(), ", input_names size",
                       input_names.s_size()));
  TFNN_ASSERT(input_dtypes.type_size() == input_names.s_size(),
              errors::Internal("incorrect input metadata, input_dtypes size ",
                               input_dtypes.type_size(), ", input_names size",
                               input_names.s_size()));
  TFNN_ASSERT(input_shapes.shape_size() == input_names.s_size(),
              errors::Internal("incorrect input metadata, input_shapes size ",
                               input_shapes.shape_size(), ", input_names size",
                               input_names.s_size()));
  for (auto idx = 0; idx < input_dtypes.type_size(); ++idx) {
    const Tensor& in_tensor = input_tensors.at(idx);
    DataType dtype = in_tensor.dtype();
    TensorShape shape = in_tensor.shape();
    DataType dtype_expected = input_dtypes.type(idx);
    TensorShape shape_expected = input_shapes.shape(idx);
    TFNN_ASSERT(dtype == dtype_expected,
                errors::Internal("incorrect input tensor dtype ", dtype,
                                 ", expected ", dtype_expected));
    TFNN_ASSERT(shape == shape_expected,
                errors::Internal("incorrect input tensor shape ", shape,
                                 ", expected ", shape_expected));
  }
  return Status::OK();
}

static Status setup_runtime_io(RuntimeIO* runtime_io, const NodeDef& node_def,
                               const std::vector<Tensor>& input_shm_tensors,
                               const std::vector<Tensor*>& output_shm_tensors,
                               uint32_t nn_id,
                               SharedMemoryAllocator* shm_allocator,
                               bool use_shm) {
  const google::protobuf::Map<std::string, AttrValue>& attr = node_def.attr();
  AttrList& input_names = attr.at("input_names").list();
  AttrList& output_names = attr.at("output_names").list();
  std::vector<StringPiece> input_paths;
  std::vector<StringPiece> output_paths;
  if (TF_PREDICT_TRUE(use_shm)) {
    input_paths.reserve(input_shm_tensors.size());
    for (const Tensor& shm_tensor : input_shm_tensors) {
      SharedMemoryPtr shm_buf = shm_allocator->get_shm_ptr(shm_tensor);
      input_paths.push_back(shm_buf->get_path());
    }
    output_paths.reserve(output_shm_tensors.size());
    for (const Tensor* shm_tensor : output_shm_tensors) {
      SharedMemoryPtr shm_buf = shm_allocator->get_shm_ptr(*shm_tensor);
      output_paths.push_back(shm_buf->get_path());
    }
  }
  return runtime_io->setup(input_names, output_names, nn_id, use_shm,
                           input_paths, output_paths);
}

static Status copy_input_tensors(RuntimeIO* runtime_io,
                                 const std::vector<Tensor>& input_tensors,
                                 std::vector<Tensor>* input_shm_tensors,
                                 thread::ThreadPool* thread_pool) {
  if (TF_PREDICT_TRUE(runtime_io->use_shm())) {
    CHECK_VALID_PTR(input_shm_tensors);
    CHECK_SIZES_MATCH(input_shm_tensors->size(), input_tensors.size());
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      Tensor* dst = &input_shm_tensors->at(idx);
      TF_RETURN_IF_ERROR(tensor_copy(dst, input_tensors.at(idx), thread_pool));
    }
  }
  return runtime_io->copy_input_tensors(input_tensors);
}

static Status copy_input_tensors(RuntimeIO* runtime_io,
                                 const std::vector<Tensor>& input_tensors,
                                 AttrList& input_shuffles,
                                 std::vector<Tensor>* shuffle_buffers,
                                 std::vector<Tensor>* input_shm_tensors) {
  uint64 start_timestamp = Env::Default()->NowMicros();
  CHECK_SIZES_MATCH(input_shuffles.tensor_size(), input_tensors.size());
  if (TF_PREDICT_TRUE(runtime_io->use_shm())) {
    CHECK_VALID_PTR(input_shm_tensors);
    CHECK_SIZES_MATCH(input_shm_tensors->size(), input_tensors.size());
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      Tensor* dst = &input_shm_tensors->at(idx);
      const TensorProto& shuffle = input_shuffles.tensor(idx);
      TF_RETURN_IF_ERROR(tensor_shuffle(dst, input_tensors.at(idx), shuffle));
    }
    TF_RETURN_IF_ERROR(runtime_io->copy_input_tensors(input_tensors));
  } else {
    if (TF_PREDICT_FALSE(nullptr == shuffle_buffers)) {
      return errors::Internal(
          "need allocated input shuffle buffers for non-shared-memory");
    }
    if (TF_PREDICT_FALSE(shuffle_buffers->size() != input_tensors.size())) {
      return errors::Internal(
          "wrong number of allocated input shuffle buffers");
    }
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      Tensor* dst = &shuffle_buffers->at(idx);
      const TensorProto& shuffle = input_shuffles.tensor(idx);
      TF_RETURN_IF_ERROR(tensor_shuffle(dst, input_tensors.at(idx), shuffle));
    }
    TF_RETURN_IF_ERROR(runtime_io->copy_input_tensors(*shuffle_buffers));
  }
  uint64 elapsed = Env::Default()->NowMicros() - start_timestamp;
  VLOG(1) << "input copy and shuffle for " << input_tensors.size()
          << " tensors took " << elapsed << " us";
  return Status::OK();
}

static Status copy_input_tensors_with_shuffle(
    OpKernelContext* ctx, const NodeDef& node_def,
    thread::ThreadPool* thread_pool, const std::vector<Tensor>& input_tensors,
    RuntimeIO* runtime_io, std::vector<Tensor>* input_shm_tensors) {
  const google::protobuf::Map<std::string, AttrValue>& attr = node_def.attr();
  if (attr.count("_input_shuffles")) {
    AttrList& input_shuffles = attr.at("_input_shuffles").list();
    std::vector<Tensor> buffers;
    if (TF_PREDICT_FALSE(!runtime_io->use_shm())) {
      buffers.resize(input_tensors.size());
      for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
        const Tensor& src = input_tensors.at(idx);
        Tensor* dst = &buffers.at(idx);
        TF_RETURN_IF_ERROR(ctx->allocate_temp(src.dtype(), src.shape(), dst));
      }
    }
    RIE_IGNORE_ABORTED(copy_input_tensors(runtime_io, input_tensors,
                                          input_shuffles, &buffers,
                                          input_shm_tensors));
  } else {
    RIE_IGNORE_ABORTED(copy_input_tensors(runtime_io, input_tensors,
                                          input_shm_tensors, thread_pool));
  }
  return Status::OK();
}

NeuronModel::NeuronModel()
    : h2d_transfer_pool_(Env::Default(), "neuron_h2d", H2D_POOL_SIZE) {
  VLOG(1) << "NeuronModel contructor " << this;
}

Status NeuronModel::initialize(const NodeDef& node_def,
                               const std::string& session_handle) {
  tensorflow::mutex_lock lock(mutex_model_);
  if (TF_PREDICT_TRUE(nullptr != neuron_engine_)) {
    VLOG(1) << "NeuronModel is already initialized";
    return Status::OK();
  }
  const google::protobuf::Map<std::string, AttrValue>& attr = node_def.attr();

  // validate input shuffles
  if (attr.count("_input_shuffles")) {
    AttrList& input_shuffles = attr.at("_input_shuffles").list();
    AttrList& input_shapes = attr.at("input_shapes").list();
    if (input_shuffles.tensor_size() != input_shapes.shape_size()) {
      return errors::InvalidArgument(
          "_input_shuffles size does not agree with input_shapes");
    }
    for (auto input_idx = 0; input_idx < input_shuffles.tensor_size();
         ++input_idx) {
      int64 num_elements =
          TensorShape(input_shapes.shape(input_idx)).num_elements();
      const TensorProto& shuffle = input_shuffles.tensor(input_idx);
      for (auto idx = 0; idx < num_elements; ++idx) {
        int shuffle_idx = shuffle.int64_val(idx);
        if (!(0 <= shuffle_idx && shuffle_idx < num_elements)) {
          return errors::InvalidArgument("invalid shuffle index ", shuffle_idx);
        }
      }
    }
  }

  // validate executable
  StringPiece executable(attr.at("executable").s());
  if (executable.empty()) {
    return errors::InvalidArgument("Neuron executable (neff) is empty.");
  }
  profile_.initialize(env_get("NEURON_PROFILE"), node_def.name());
  if (profile_.enabled_)
    profile_.dump_info(attr.at("graph_def").s(), executable);
  AttrList& model_config_attr = attr.at("model_config").list();
  NeuronModelConfig model_config;
  model_config.parse_opt_engine_size(model_config_attr);
  model_config.parse_engine_index(model_config_attr);
  TF_RETURN_IF_ERROR(
      NeuronEngineManager::GetNeuronEngineManager().apply_for_engine(
          &neuron_engine_, session_handle, model_config.opt_engine_size_,
          model_config.max_num_duplicates_, model_config.engine_index_));
  model_config.parse_timeout(model_config_attr);
  model_config.parse_ninfer(model_config_attr, neuron_engine_->num_cores(),
                            NeuronEngineManager::MIN_NUM_CORES,
                            NeuronEngineManager::MAX_NUM_CORES);
  estimated_cost_ = executable.size();
  TF_RETURN_IF_ERROR(
      neuron_engine_->load(&nn_id_, executable, model_config.timeout_,
                           model_config.ninfer_, profile_.enabled_));
  VLOG(1) << "loaded " << node_def.name() << " as " << nn_id_
          << "; number of NEFFs: " << neuron_engine_->num_executable();

  // check argument sizes
  TF_RETURN_IF_ERROR(get_io_tensor_sizes(nullptr, node_def, "input"));
  TF_RETURN_IF_ERROR(get_io_tensor_sizes(nullptr, node_def, "output"));
  return Status::OK();
}

Status NeuronModel::compute(OpKernelContext* ctx, const NodeDef& node_def,
                            const std::vector<Tensor>& input_tensors) {
  uint64 start_time = Env::Default()->NowMicros();
#define VLOG_TIME(msg) VLOG_TIME_BASE(start_time, 1, msg);
  SharedMemoryAllocator* shm_allocator =
      NeuronEngineManager::GetNeuronEngineManager().get_shm_allocator();
  thread::ThreadPool* thread_pool =
      ctx->device()->tensorflow_cpu_worker_threads()->workers;
  const google::protobuf::Map<std::string, AttrValue>& attr = node_def.attr();
  AttrList& input_names = attr.at("input_names").list();
  AttrList& output_shapes = attr.at("output_shapes").list();
  TFNN_ASSERT((int)input_tensors.size() == input_names.s_size(),
              errors::InvalidArgument("incorrect number of input tensors"));
  if (attr.count("_input_shuffles")) {
    AttrList& input_shuffles = attr.at("_input_shuffles").list();
    TFNN_ASSERT(input_shuffles.tensor_size() == (int64)input_tensors.size(),
                errors::InvalidArgument("illegal _input_shuffles attribute"));
  }
   std::vector<size_t> output_tensor_sizes;
  TF_RETURN_IF_ERROR(
      get_io_tensor_sizes(&output_tensor_sizes, node_def, "output"));

  // enable/disable dynamic batch size
  int64_t batch_size = UNINIT_BATCH_SIZE;
  int64_t k_batch_size = UNINIT_BATCH_SIZE;
  std::vector<bool> is_batch_inputs(input_tensors.size());
  std::vector<bool> is_batch_outputs(ctx->num_outputs());
  AttrList& output_names = attr.at("output_names").list();
  AttrList& input_batch_axis = attr.at("input_batch_axis").list();
  AttrList& output_batch_axis = attr.at("output_batch_axis").list();
  bool found_batch_axis = false;
  if (TF_PREDICT_TRUE(input_names.s_size() == input_batch_axis.i_size())) {
    if (TF_PREDICT_TRUE(output_names.s_size() == output_batch_axis.i_size())) {
      for (int64 batch_axis : input_batch_axis.i()) {
        if (batch_axis != STATIC_BATCH_AXIS) {
          found_batch_axis = true;
          break;
        }
      }
    }
  }
  bool use_dynamic_batch_size = false;
  int64 input_copy_cost_per_unit = 0;
  if (found_batch_axis) {
    AttrList& input_shapes = attr.at("input_shapes").list();
    for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
      bool is_batch_tensor = false;
      const Tensor& in_tensor = input_tensors.at(idx);
      TensorShape shape(in_tensor.shape());
      TensorShape k_shape(input_shapes.shape(idx));
      input_copy_cost_per_unit += k_shape.num_elements();
      if (TF_PREDICT_TRUE(0 == input_batch_axis.i(idx))) {
        TFNN_ASSERT(
            shape.dims() > 0,
            errors::InvalidArgument("no batch-dimension found on input tensor ",
                                    input_names.s(idx), " with shape ",
                                    shape.DebugString()));
        if (TF_PREDICT_TRUE(UNINIT_BATCH_SIZE == batch_size)) {
          batch_size = shape.dim_size(0);
          k_batch_size = k_shape.dim_size(0);
          TFNN_ASSERT(
              batch_size > 0,
              errors::Internal(
                  "incorrect internal batch size inferred from input tensor ",
                  input_names.s(idx), " with shape ", shape.DebugString()));
        } else {
          TFNN_ASSERT(
              batch_size == shape.dim_size(0),
              errors::InvalidArgument(
                  "incorrect batch size found on input tensor ",
                  input_names.s(idx), ", tensor shape ", shape.DebugString(),
                  ", internal batch size ", batch_size));
        }
        shape.RemoveDim(0);
        k_shape.RemoveDim(0);
        is_batch_tensor = batch_size != k_batch_size;
        use_dynamic_batch_size |= is_batch_tensor;
      }
      TFNN_ASSERT(
          shape == k_shape,
          errors::InvalidArgument(
              "incorrect shape found on input tensor ", input_names.s(idx),
              ", inference time shape ", in_tensor.shape().DebugString(),
              ", expected shape ", input_shapes.shape(idx).DebugString()));
      is_batch_inputs[idx] = is_batch_tensor;
    }
    for (auto idx = 0; idx < output_names.s_size(); ++idx) {
      bool is_batch_tensor = false;
      if (TF_PREDICT_TRUE(0 == output_batch_axis.i(idx))) {
        TensorShape k_shape(output_shapes.shape(idx));
        TFNN_ASSERT(k_shape.dims() > 0,
                    errors::InvalidArgument(
                        "no batch-dimension found on output tensor ",
                        output_names.s(idx), " with Neuron shape ",
                        k_shape.DebugString()));
        TFNN_ASSERT(
            k_batch_size == k_shape.dim_size(0),
            errors::InvalidArgument(
                "incorrect batch size found on output tensor ",
                output_names.s(idx), ", Neuron tensor shape ",
                k_shape.DebugString(), ", Neuron batch size ", k_batch_size));
        is_batch_tensor = batch_size != k_shape.dim_size(0);
        use_dynamic_batch_size |= is_batch_tensor;
      }
      is_batch_outputs[idx] = is_batch_tensor;
    }
  }
  TFNN_ASSERT(ctx->num_outputs() == output_names.s_size(),
              errors::InvalidArgument("incorrect number of output tensors"));

  // allocate output tensors
  std::vector<Tensor*> output_tensors(ctx->num_outputs());
  int64_t pad_batch_size = 0;
  if (use_dynamic_batch_size) {
    pad_batch_size = ((batch_size - 1) / k_batch_size + 1) * k_batch_size;
    VLOG(1) << "batch_size=" << batch_size << ", k_batch_size=" << k_batch_size
            << ", pad_batch_size=" << pad_batch_size;
    for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
      Tensor* batch_out_tensor = nullptr;
      TensorShape shape(output_shapes.shape(idx));
      if (TF_PREDICT_TRUE(is_batch_outputs[idx])) {
        shape.set_dim(0, batch_size);
      }
      TF_RETURN_IF_ERROR(ctx->allocate_output(idx, shape, &batch_out_tensor));
      output_tensors[idx] = batch_out_tensor;
    }
  } else {
    for (auto idx = 0; idx < ctx->num_outputs(); ++idx) {
      TF_RETURN_IF_ERROR(ctx->allocate_output(idx, output_shapes.shape(idx),
                                              &output_tensors[idx]));
    }
  }

  // initialize the model
  RIE_IGNORE_ABORTED(initialize(node_def, ctx->session_handle()));

  // keep a shared pointer so that RuntimeSession outlives shared memory buffers
  std::shared_ptr<RuntimeSession> session_alive = neuron_engine_->get_session();

  // run inference
  if (use_dynamic_batch_size) {
    int64 end_start = k_batch_size - (pad_batch_size - batch_size);
    bool run_profiler_in_shard = false;
    Status status_sd;
#define SHARD_LOG_ERROR(status_sd, ...)                            \
  {                                                                \
    Status _status = (__VA_ARGS__);                                \
    if (TF_PREDICT_FALSE(!_status.ok())) {                         \
      LOG(ERROR) << "shard error code " << _status.code()          \
                 << ", error message " << _status.error_message(); \
      status_sd = _status;                                         \
      return;                                                      \
    }                                                              \
  }
#define SHARD_LOG_IGNORE_ABORTED(status_sd, ...)                      \
  {                                                                   \
    Status _status(__VA_ARGS__);                                      \
    if (TF_PREDICT_FALSE(                                             \
            !(_status.ok() ||                                         \
              _status.code() == tensorflow::error::Code::ABORTED))) { \
      LOG(ERROR) << "shard error code " << _status.code()             \
                 << ", error message " << _status.error_message();    \
      status_sd = _status;                                            \
      return;                                                         \
    }                                                                 \
  }
#define SHARD_VLOG_TIME(msg) VLOG_TIME_BASE(start_time, 2, msg);
    auto ShardFunc = [&](int64 dim0_start, int64 dim0_limit) {
      SHARD_VLOG_TIME("entering shard");
      if (TF_PREDICT_FALSE(dim0_limit - dim0_start != k_batch_size)) {
        status_sd =
            errors::Internal("illegal shard ", dim0_start, ":", dim0_limit);
        return;
      }
      VLOG(2) << "Sharding " << dim0_start << " to " << dim0_limit;
      std::vector<Tensor> sliced_inputs(input_tensors.size());
      for (size_t idx = 0; idx < input_tensors.size(); ++idx) {
        const Tensor& in_tensor = input_tensors.at(idx);
        if (TF_PREDICT_TRUE(is_batch_inputs[idx])) {
          if (TF_PREDICT_FALSE(dim0_limit > batch_size)) {
            TensorShape ps_shape(in_tensor.shape());
            ps_shape.set_dim(0, k_batch_size);
            Tensor pad_end_slice(in_tensor.dtype(), ps_shape);
            Tensor zero_slice = pad_end_slice.Slice(end_start, k_batch_size);
            SHARD_LOG_ERROR(status_sd, tensor_memset(&zero_slice, 0));
            Tensor end_slice = in_tensor.Slice(dim0_start, batch_size);
            SHARD_LOG_ERROR(status_sd, tensor_copy(&pad_end_slice, end_slice));
            sliced_inputs[idx] = pad_end_slice;
          } else {
            sliced_inputs[idx] = in_tensor.Slice(dim0_start, dim0_limit);
          }
        } else {
          sliced_inputs[idx] = in_tensor;
        }
      }
      SHARD_LOG_ERROR(status_sd, check_input_tensors(sliced_inputs, node_def));
      int64 end_limit = dim0_limit < batch_size ? dim0_limit : batch_size;
      std::vector<Tensor> sliced_outputs(output_tensors.size());
      for (size_t idx = 0; idx < sliced_outputs.size(); ++idx) {
        Tensor* out_tensor = output_tensors.at(idx);
        if (TF_PREDICT_TRUE(is_batch_outputs[idx])) {
          sliced_outputs[idx] = out_tensor->Slice(dim0_start, end_limit);
        } else {
          sliced_outputs[idx] = *out_tensor;
        }
      }
      std::vector<Tensor*> output_ptrs(sliced_outputs.size());
      for (size_t idx = 0; idx < output_ptrs.size(); ++idx) {
        output_ptrs[idx] = &sliced_outputs.at(idx);
      }
      RuntimeIO runtime_io;
      std::vector<Tensor> input_shm_tensors;
      std::vector<Tensor> output_shm_tensors;
      bool use_shm = shm_allocator->is_valid();
      for (const Tensor& tensor : sliced_inputs) {
        use_shm &= tensor.NumElements() != 0;
      }
      for (size_t buf_size : output_tensor_sizes) {
        use_shm &= buf_size != 0;
      }
      if (TF_PREDICT_TRUE(use_shm)) {
        input_shm_tensors.resize(sliced_inputs.size());
        for (size_t idx = 0; idx < sliced_inputs.size(); ++idx) {
          const Tensor& tensor = sliced_inputs.at(idx);
          TensorShape shape = tensor.shape();
          DataType dtype = tensor.dtype();
          AllocatorAttributes attr;
          NeuronDevice::set_on_shm(&attr, true);
          Tensor& shm_tensor = input_shm_tensors.at(idx);
          SHARD_LOG_ERROR(status_sd,
                          ctx->allocate_temp(dtype, shape, &shm_tensor, attr));
        }
        output_shm_tensors.resize(sliced_outputs.size());
        for (size_t idx = 0; idx < output_shm_tensors.size(); ++idx) {
          const Tensor& tensor = sliced_outputs.at(idx);
          TensorShape shape(tensor.shape());
          if (TF_PREDICT_TRUE(is_batch_outputs[idx])) {
            if (TF_PREDICT_FALSE(dim0_limit > batch_size)) {
              shape.set_dim(0, k_batch_size);
            }
          }
          DataType dtype(tensor.dtype());
          AllocatorAttributes attr;
          NeuronDevice::set_on_shm(&attr, true);
          Tensor& shm_tensor = output_shm_tensors.at(idx);
          SHARD_LOG_ERROR(status_sd,
                          ctx->allocate_temp(dtype, shape, &shm_tensor, attr));
        }
      }
      std::vector<Tensor*> output_shm_ptrs;
      for (Tensor& shm_tensor : output_shm_tensors) {
        output_shm_ptrs.push_back(&shm_tensor);
      }
      SHARD_LOG_IGNORE_ABORTED(
          status_sd, setup_runtime_io(&runtime_io, node_def, input_shm_tensors,
                                      output_shm_ptrs, nn_id_, shm_allocator,
                                      use_shm));

      // copy input tensors with optional input_shuffles
      SHARD_VLOG_TIME("in shard before input copy");
      if (k_batch_size > 1 && runtime_io.use_shm()) {
        auto CopyInputShardFunc = [&](int64 dim0_start, int64 dim0_limit) {
          std::vector<Tensor> input_slices(sliced_inputs.size());
          for (size_t i = 0; i < input_slices.size(); ++i) {
            if (TF_PREDICT_TRUE(is_batch_inputs[i])) {
              input_slices[i] = sliced_inputs[i].Slice(dim0_start, dim0_limit);
            } else {
              input_slices[i] = input_tensors[i];
            }
          }
          std::vector<Tensor> input_shm_slices(sliced_inputs.size());
          for (size_t i = 0; i < input_shm_slices.size(); ++i) {
            Tensor& shm_tensor = input_shm_tensors.at(i);
            if (TF_PREDICT_TRUE(is_batch_inputs[i])) {
              input_shm_slices[i] = shm_tensor.Slice(dim0_start, dim0_limit);
            } else {
              input_shm_slices[i] = shm_tensor;
            }
          }
          SHARD_LOG_IGNORE_ABORTED(
              status_sd, copy_input_tensors_with_shuffle(
                             ctx, node_def, nullptr, input_slices, &runtime_io,
                             &input_shm_slices));
        };
        h2d_transfer_pool_.ParallelFor(k_batch_size, input_copy_cost_per_unit,
                                       std::move(CopyInputShardFunc));
      } else {
        SHARD_LOG_IGNORE_ABORTED(
            status_sd, copy_input_tensors_with_shuffle(
                           ctx, node_def, &h2d_transfer_pool_, sliced_inputs,
                           &runtime_io, &input_shm_tensors));
      }

      // run inference
      SHARD_VLOG_TIME("in shard before infer");
      if (TF_PREDICT_FALSE(run_profiler_in_shard)) {
        VLOG(1) << "enabling profiler in shard";
        SHARD_LOG_IGNORE_ABORTED(
            status_sd,
            neuron_engine_->infer_with_profiling(&runtime_io, &profile_));
      } else {
        SHARD_LOG_IGNORE_ABORTED(status_sd, neuron_engine_->infer(&runtime_io));
      }
      SHARD_VLOG_TIME("in shard after infer");
      SHARD_LOG_IGNORE_ABORTED(
          status_sd, runtime_io.finish(&output_ptrs, output_shm_tensors,
                                       &h2d_transfer_pool_));
      SHARD_VLOG_TIME("in shard exit");
    };
#undef SHARD_LOG_IGNORE_ABORTED
#undef SHARD_LOG_ERROR
#undef SHARD_VLOG_TIME
    if (TF_PREDICT_FALSE(profile_.enabled_)) {
      run_profiler_in_shard = true;
      ShardFunc(0, k_batch_size);
      run_profiler_in_shard = false;
      RIE_IGNORE_ABORTED(status_sd);
    }
    VLOG_TIME("before sharding");
#if TF_VERSION_LESS_THAN(2, 0)
    thread_pool->TransformRangeConcurrently(k_batch_size, pad_batch_size,
                                            std::move(ShardFunc));
#else
    auto strategy = thread::ThreadPool::SchedulingStrategy::kFixedBlockSize;
    int64 cost_per_unit = estimated_cost_;
    auto params = thread::ThreadPool::SchedulingParams(strategy, cost_per_unit,
                                                       k_batch_size);
    thread_pool->ParallelFor(pad_batch_size, params, std::move(ShardFunc));
#endif
    RIE_IGNORE_ABORTED(status_sd);
  } else {
    TF_RETURN_IF_ERROR(check_input_tensors(input_tensors, node_def));
    RuntimeIO runtime_io;
    std::vector<Tensor> input_shm_tensors;
    std::vector<Tensor> output_shm_tensors;
    bool use_shm = shm_allocator->is_valid();
    for (const Tensor& tensor : input_tensors) {
      use_shm &= tensor.NumElements() != 0;
    }
    for (size_t buf_size : output_tensor_sizes) {
      use_shm &= buf_size != 0;
    }
    if (TF_PREDICT_TRUE(use_shm)) {
      input_shm_tensors.resize(input_tensors.size());
      for (size_t idx = 0; idx < input_shm_tensors.size(); ++idx) {
        const Tensor& tensor = input_tensors.at(idx);
        TensorShape shape = tensor.shape();
        DataType dtype = tensor.dtype();
        AllocatorAttributes attr;
        NeuronDevice::set_on_shm(&attr, true);
        Tensor& shm_tensor = input_shm_tensors.at(idx);
        TF_RETURN_IF_ERROR(ctx->allocate_temp(dtype, shape, &shm_tensor, attr));
      }
      output_shm_tensors.resize(output_tensors.size());
      for (size_t idx = 0; idx < output_shm_tensors.size(); ++idx) {
        Tensor* tensor = output_tensors.at(idx);
        TensorShape shape(tensor->shape());
        DataType dtype(tensor->dtype());
        AllocatorAttributes attr;
        NeuronDevice::set_on_shm(&attr, true);
        Tensor& shm_tensor = output_shm_tensors.at(idx);
        TF_RETURN_IF_ERROR(ctx->allocate_temp(dtype, shape, &shm_tensor, attr));
      }
    }
    std::vector<Tensor*> output_shm_ptrs;
    for (Tensor& shm_tensor : output_shm_tensors) {
      output_shm_ptrs.push_back(&shm_tensor);
    }
    RIE_IGNORE_ABORTED(setup_runtime_io(&runtime_io, node_def,
                                        input_shm_tensors, output_shm_ptrs,
                                        nn_id_, shm_allocator, use_shm));

    // copy input tensors with optional input_shuffles
    RIE_IGNORE_ABORTED(copy_input_tensors_with_shuffle(
        ctx, node_def, thread_pool, input_tensors, &runtime_io,
        &input_shm_tensors));

    // run inference
    VLOG_TIME("before infer");
    if (TF_PREDICT_FALSE(profile_.enabled_)) {
      VLOG(1) << "profile enabled -- lock stop/start/infer altogether";
      RIE_IGNORE_ABORTED(
          neuron_engine_->infer_with_profiling(&runtime_io, &profile_));
    } else {
      RIE_IGNORE_ABORTED(neuron_engine_->infer(&runtime_io));
    }
    VLOG_TIME("after infer");
    RIE_IGNORE_ABORTED(
        runtime_io.finish(&output_tensors, output_shm_tensors, thread_pool));
  }
  VLOG_TIME("exiting compute");
#undef VLOG_TIME
  return Status::OK();
}

NeuronModel::~NeuronModel() {
  VLOG(1) << "calling NeuronModel destructor";
  tensorflow::mutex_lock lock(mutex_model_);
  if (nullptr == neuron_engine_) {
    VLOG(1) << "neuron_engine_ not available; not tearing down";
    return;
  }
  neuron_engine_->unload(nn_id_);
  VLOG(1) << "unload from NeuronModel::~NeuronModel";
  NeuronEngineManager::GetNeuronEngineManager().clear_if_empty();
  VLOG(1) << "NeuronModel destructor done";
}

}  // namespace neuron
}  // namespace tensorflow

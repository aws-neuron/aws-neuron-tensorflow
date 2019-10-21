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
#include "tensorflow/python/neuron/kernels/inferentia_op.h"
#include "tensorflow/python/neuron/util/logging.h"
#include <grpcpp/grpcpp.h>


namespace tensorflow {
namespace kaena {


static const size_t EXEC_MAX_CHUNK_SIZE = 1024 * 1024;  // some reasonable number of bytes
static const int64 UNINIT_BATCH_SIZE = -8;  // magic number for uninitialized batch size


TPBManager global_tpb_manager;


InferentiaOp::InferentiaOp(OpKernelConstruction *context) : OpKernel(context) {
    // read executable
    std::string executable;
    OP_REQUIRES_OK(context, context->GetAttr("executable", &executable));
    if ("" != executable) {
        op_name_ = context->def().name();
        std::vector<std::string> input_names;
        OP_REQUIRES_OK(context, context->GetAttr("input_names", &input_names));
        OP_REQUIRES_OK(context, context->GetAttr("input_dtypes", &input_dtypes_));
        OP_REQUIRES_OK(context, context->GetAttr("input_shapes", &input_shapes_));
        OP_REQUIRES_OK(context, context->GetAttr("input_batch_axis", &input_batch_axis_));
        OP_REQUIRES_OK(context, context->GetAttr("output_batch_axis", &output_batch_axis_));
        std::vector<std::string> output_names;
        OP_REQUIRES_OK(context, context->GetAttr("output_names", &output_names));
        std::vector<DataType> output_dtypes;
        OP_REQUIRES_OK(context, context->GetAttr("output_dtypes", &output_dtypes));
        std::vector<TensorShape> output_shapes;
        OP_REQUIRES_OK(context, context->GetAttr("output_shapes", &output_shapes));
        tensorflow::Status status = initialize(
            executable, input_names, output_names, output_dtypes, output_shapes);
        if (!status.ok()) INFERENTIA_OP_ERROR(context, status);
    }
    if (profile_enabled_) {
        std::string graph_def;
        OP_REQUIRES_OK(context, context->GetAttr("graph_def", &graph_def));
        profile_dump_info(graph_def, executable);
    }
}


tensorflow::Status InferentiaOp::initialize(
        const std::string &executable,
        const std::vector<std::string> &input_names,
        const std::vector<std::string> &output_names,
        const std::vector<DataType> &output_dtypes,
        const std::vector<TensorShape> &output_shapes) {
    krtd_server_ = env_get("NEURON_RTD_ADDRESS", "unix:/run/neuron.sock");

    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    ch_args.SetMaxSendMessageSize(-1);
    std::shared_ptr<grpc::Channel> krtd_channel = grpc::CreateCustomChannel(
        krtd_server_, grpc::InsecureChannelCredentials(), ch_args);
    if (nullptr == krtd_channel) {
        KAENA_ERROR_STATUS("cannot establish grpc channel to krtd server");
    }
    stub_ = krt::kmgr_v1::NewStub(krtd_channel);
    if (nullptr == stub_) {
        KAENA_ERROR_STATUS("cannot create krtd stub");
    }

    if (!global_tpb_manager.ready()) {
        tensorflow::Status status = global_tpb_manager.initialize();
        if (!status.ok()) {
            return status;
        }
    }
    if (!global_tpb_manager.ready()) {
        KAENA_ERROR_STATUS("global_tpb_manager initialization failure");
    }

    // create_eg
    grpc::Status status;
    tpb_group_ = global_tpb_manager.get_tpb_group();

    // load
    grpc::ClientContext context;
    krt::load_response load_response;
    std::unique_ptr<grpc::ClientWriter<krt::load_request> > writer(
        stub_->load(&context, &load_response));
    krt::load_request load_request;

    // krt_eg_id
    load_request.mutable_h_eg()->set_id(tpb_group_->get_krt_eg_id());
    writer->Write(load_request);

    // kelp_size
    size_t exec_total_size = executable.size();
    load_request.set_kelp_size(exec_total_size);
    writer->Write(load_request);

    // todo: read these info from kelp
    infer_timeout_ = 10;
    infer_queue_length_ = 4;
    krt::model_params *model_params = load_request.mutable_model_params();
    model_params->mutable_timeout()->set_data(infer_timeout_);
    model_params->mutable_ninfer()->set_data(infer_queue_length_);
    writer->Write(load_request);

    // kelp file content
    StringPiece executable_view(executable);
    for (size_t pos = 0; pos < exec_total_size; pos += EXEC_MAX_CHUNK_SIZE) {
        size_t remaining = exec_total_size - pos;
        size_t chunk_size = std::min(remaining, EXEC_MAX_CHUNK_SIZE);
        StringPiece file_chunk = executable_view.substr(pos, chunk_size);
        load_request.mutable_kelp_chunk()->set_chunk(file_chunk.data(), chunk_size);
        writer->Write(load_request);
    }
    writer->WritesDone();
    status = writer->Finish();
    KRTD_CHECK_RETURN("load", status, load_response);
    krt_nn_id_ = load_response.h_nn().id();
    krt_load_done_ = true;
    tpb_group_->register_executable(krt_nn_id_);
    KAENALOG(1) << "load: number of executables: " << tpb_group_->get_num_executable();

    // check argument sizes
    if (input_names.size() != input_dtypes_.size()
            || input_names.size() != input_shapes_.size()) {
        KAENA_ERROR_STATUS(
            "incorrect number of inputs: input_names size ", input_names.size(),
            ", input_dtypes size ", input_dtypes_.size(),
            ", input_shapes size ", input_shapes_.size());
    }
    if (output_names.size() != output_dtypes.size()
            || output_names.size() != output_shapes.size()) {
        KAENA_ERROR_STATUS(
            "incorrect number of outputs: output_names size ", output_names.size(),
            ", output_dtypes size ", output_dtypes.size(),
            ", output_shapes size ", output_shapes.size());
    }
    for (size_t idx = 0; idx < input_dtypes_.size(); ++idx) {
        Tensor temp_tensor(input_dtypes_[idx], input_shapes_[idx]);
        input_tensor_sizes_.push_back(temp_tensor.tensor_data().size());
    }

    for (auto &name : input_names) {
        input_names_.push_back(name);
    }
    for (auto &name : output_names) {
        output_names_.push_back(name);
    }
    // prepare output tensors
    if (0 == krtd_server_.find("unix:")
            && prepare_shared_memory(output_dtypes, output_shapes).ok()) {
        for (size_t idx = 0; idx < output_dtypes.size(); ++idx) {
            output_tensors_.emplace_back(
                &output_shm_allocs_[idx], output_dtypes[idx], output_shapes[idx]);
        }
        use_shared_memory_ = true;
    } else {
        for (size_t idx = 0; idx < output_dtypes.size(); ++idx) {
            output_tensors_.emplace_back(output_dtypes[idx], output_shapes[idx]);
        }
    }
    profile_dir_ = env_get("NEURON_PROFILE");
    profile_enabled_ = "" != profile_dir_;
    ready_ = true;
    return tensorflow::Status::OK();
}


tensorflow::Status InferentiaOp::prepare_shared_memory(
        const std::vector<DataType> &output_dtypes,
        const std::vector<TensorShape> &output_shapes) {
    for (size_t idx = 0; idx < input_tensor_sizes_.size(); ++idx) {
        size_t shm_size = input_tensor_sizes_[idx];
        input_shms_.emplace_back(shm_size);
        tensorflow::Status tf_status = input_shms_.back().initialize(stub_);
        if (!tf_status.ok()) {
            return tf_status;
        }
        KAENALOG(1) << "input shared memory " << input_shms_.back().name()
                    << " ready at address " << input_shms_.back().ptr();
    }
    for (size_t idx = 0; idx < output_dtypes.size(); ++idx) {
        Tensor temp_tensor(output_dtypes[idx], output_shapes[idx]);
        size_t shm_size = temp_tensor.tensor_data().size();
        output_shms_.emplace_back(shm_size);
        tensorflow::Status tf_status = output_shms_.back().initialize(stub_);
        if (!tf_status.ok()) {
            return tf_status;
        }
        KAENALOG(1) << "output shared memory " << output_shms_.back().name()
                    << " ready at address " << output_shms_.back().ptr();
    }
    for (auto &out_shm : output_shms_) {
        tpb_group_->get_ptr2shm()->emplace(out_shm.ptr(), &out_shm);
    }
    for (auto &out_shm : output_shms_) {
        output_shm_allocs_.emplace_back(&out_shm);
    }
    return tensorflow::Status::OK();
}


InferentiaOp::~InferentiaOp() {
    if (nullptr == tpb_group_) {
        KAENA_ERROR("tpb_group_ not available; not tearing down");
        return;
    }
    grpc::Status status;

    // stop
    if (tpb_group_->nn_is_running(krt_nn_id_)) {
        grpc::ClientContext context;
        krt::stop_request stop_request;
        stop_request.mutable_h_nn()->set_id(krt_nn_id_);
        krt::stop_response stop_response;
        status = stub_->stop(&context, stop_request, &stop_response);
        KRTD_CHECK("stop", status, stop_response);
        tpb_group_->nn_set_current_running(KRT_INVALID_NN_ID);
    }

    // unload
    if (krt_load_done_) {
        grpc::ClientContext context;
        krt::unload_request unload_request;
        unload_request.mutable_h_nn()->set_id(krt_nn_id_);
        krt::unload_response unload_response;
        status = stub_->unload(&context, unload_request, &unload_response);
        KRTD_CHECK("unload", status, unload_response);
    }
    tpb_group_->deregister_executable(krt_nn_id_);
    KAENALOG(1) << "unload: number of executables: " << tpb_group_->get_num_executable();

    // unmap all shared memories
    for (auto &shm : input_shms_) {
        shm.clear(stub_);
    }
    for (auto &shm : output_shms_) {
        tpb_group_->get_ptr2shm()->erase(shm.ptr());
        shm.clear(stub_);
    }

    // clear global_tpb_manager if it's empty -- triggers only in single-eg case
    if (global_tpb_manager.is_empty()) {
        global_tpb_manager.clear();
    }
}


static tensorflow::Status tensor_memcpy(Tensor *tensor, StringPiece &source,
                                        int64 memcpy_size=-1) {
    if (memcpy_size < 0 ? source.size() != tensor->tensor_data().size()
                        : memcpy_size > tensor->tensor_data().size()) {
        KAENA_ERROR_STATUS("unexpected tensor size in tensor_memcpy",
                           ", source size: ", source.size(),
                           ", target size: ", tensor->tensor_data().size());
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
        KAENA_ERROR_STATUS("tensor->dtype() is unsupported");
    }
    return tensorflow::Status::OK();
}


static tensorflow::Status tensor_memset(Tensor *tensor, int ch) {
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
        KAENA_ERROR_STATUS("tensor->dtype() is unsupported");
    }
    return tensorflow::Status::OK();
}


void InferentiaOp::Compute(OpKernelContext *context) {
    FALTimestamps timestamps;
    timestamps.mark_enter();

    std::vector<const Tensor*> input_tensors;
    for (auto idx = 0; idx < context->num_inputs(); ++idx) {
        input_tensors.push_back(&context->input(idx));
    }
    if (input_tensors.size() != input_names_.size()) {
        INFERENTIA_OP_ERROR(context, "incorrect number of input tensors");
    }

    int64_t batch_size = UNINIT_BATCH_SIZE;
    int64_t k_batch_size = UNINIT_BATCH_SIZE;
    std::vector<bool> is_batch_input_tensors;
    std::vector<bool> is_batch_output_tensors;
    if (input_names_.size() == input_batch_axis_.size() &&
            output_names_.size() == output_batch_axis_.size()) {
        for (auto idx = 0; idx < input_tensors.size(); ++idx) {
            bool is_batch_tensor = false;
            const Tensor *tptr = input_tensors[idx];
            TensorShape shape(tptr->shape());
            TensorShape k_shape(input_shapes_[idx]);
            if (0 == input_batch_axis_[idx]) {
                if (shape.dims() < 1) {
                    INFERENTIA_OP_ERROR(
                        context, "no batch-dimension found on input tensor ",
                        input_names_[idx], " with shape ", shape.DebugString());
                }
                if (UNINIT_BATCH_SIZE == batch_size) {
                    batch_size = shape.dim_size(0);
                    k_batch_size = k_shape.dim_size(0);
                    if (batch_size < 1) {
                        INFERENTIA_OP_ERROR(
                            context,
                            "incorrect internal batch size inferred from input tensor ",
                            input_names_[idx], " with shape ", shape.DebugString());
                    }
                } else if (batch_size != shape.dim_size(0)) {
                    INFERENTIA_OP_ERROR(
                        context, "incorrect batch size found on input tensor ",
                        input_names_[idx], ", tensor shape ", shape.DebugString(),
                        ", internal batch size ", batch_size);
                }
                shape.RemoveDim(0);
                k_shape.RemoveDim(0);
                is_batch_tensor = batch_size != k_batch_size;
            }
            if (shape != k_shape) {
                INFERENTIA_OP_ERROR(
                    context, "incorrect shape found on input tensor ", input_names_[idx],
                    ", inference time shape ", tptr->shape().DebugString(),
                    ", expected shape ", input_shapes_[idx].DebugString());
            }
            is_batch_input_tensors.push_back(is_batch_tensor);
        }
        for (auto idx = 0; idx < output_tensors_.size(); ++idx) {
            bool is_batch_tensor = false;
            if (0 == output_batch_axis_[idx]) {
                TensorShape k_shape(output_tensors_[idx].shape());
                if (k_shape.dims() < 1) {
                    INFERENTIA_OP_ERROR(
                        context, "no batch-dimension found on output tensor ",
                        output_names_[idx], " with kaena shape ", k_shape.DebugString());
                }
                if (k_batch_size != k_shape.dim_size(0)) {
                    INFERENTIA_OP_ERROR(
                        context, "incorrect batch size found on output tensor ",
                        output_names_[idx], ", kaena tensor shape ", k_shape.DebugString(),
                        ", kaena batch size ", k_batch_size);
                }
                is_batch_tensor = batch_size != k_shape.dim_size(0);
            }
            is_batch_output_tensors.push_back(is_batch_tensor);
        }
    }
    if (context->num_outputs() != output_tensors_.size()) {
        INFERENTIA_OP_ERROR(context, "incorrect number of output tensors");
    }

    tensorflow::Status status = start_model();
    if (!status.ok()) INFERENTIA_OP_ERROR(context, status);
    if (batch_size > 0) {
        int64_t pad_batch_size = ((batch_size - 1) / k_batch_size + 1) * k_batch_size;
        std::vector<Tensor*> batch_output_tensors;
        for (auto idx = 0; idx < context->num_outputs(); ++idx) {
            Tensor *batch_out_tensor = nullptr;
            if (is_batch_output_tensors[idx]) {
                TensorShape shape(output_tensors_[idx].shape());
                shape.set_dim(0, batch_size);
                context->allocate_output(idx, shape, &batch_out_tensor);
            } else {
                context->set_output(idx, output_tensors_[idx]);
            }
            batch_output_tensors.push_back(batch_out_tensor);
        }

        std::vector<std::vector<Tensor> > batches_kaena_input_tensors;
        int64_t num_batches = pad_batch_size / k_batch_size;
        for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int64_t dim0_start = batch_idx * k_batch_size;
            int64_t dim0_limit = batch_idx * k_batch_size + k_batch_size;
            batches_kaena_input_tensors.emplace_back();
            for (auto idx = 0; idx < input_tensors.size(); ++idx) {
                if (is_batch_input_tensors[idx]) {
                    if (batch_idx == num_batches - 1) {
                        TensorShape ps_shape(input_tensors[idx]->shape());
                        ps_shape.set_dim(0, k_batch_size);
                        Tensor pad_end_slice(input_tensors[idx]->dtype(), ps_shape);
                        Tensor zero_slice = pad_end_slice.Slice(
                            k_batch_size - (pad_batch_size - batch_size), k_batch_size);
                        status = tensor_memset(&zero_slice, 0);
                        if (!status.ok()) INFERENTIA_OP_ERROR(context, status);
                        Tensor end_slice = input_tensors[idx]->Slice(
                            dim0_start, batch_size);
                        StringPiece t_data = end_slice.tensor_data();
                        status = tensor_memcpy(&pad_end_slice, t_data, t_data.size());
                        if (!status.ok()) INFERENTIA_OP_ERROR(context, status);
                        batches_kaena_input_tensors.back().emplace_back(pad_end_slice);
                    } else {
                        batches_kaena_input_tensors.back().emplace_back(
                            input_tensors[idx]->Slice(dim0_start, dim0_limit));
                    }
                } else {
                    batches_kaena_input_tensors.back().emplace_back();
                }
            }
        }

        std::vector<uint64_t> infer_post_cookie_vec;
        {
            std::lock_guard<std::mutex> guard(*tpb_group_->get_mutex_infer());
            timestamps.mark_above_krtd_infer();
            int64_t start = 0;
            int64_t end = std::min(start + infer_queue_length_, num_batches);
            while (start < num_batches) {
                for (int64_t batch_idx = start; batch_idx < end; ++batch_idx) {
                    std::vector<const Tensor*> kaena_input_tensors;
                    for (auto idx = 0; idx < input_names_.size(); ++idx) {
                        if (is_batch_input_tensors[idx]) {
                            kaena_input_tensors.push_back(
                                &batches_kaena_input_tensors[batch_idx][idx]);
                        } else {
                            kaena_input_tensors.push_back(input_tensors[idx]);
                        }
                    }
                    uint64_t infer_post_cookie;
                    status = infer_post(&infer_post_cookie, kaena_input_tensors);
                    infer_post_cookie_vec.push_back(infer_post_cookie);
                    if (!status.ok()) {
                        INFERENTIA_OP_ERROR(context, status);
                    }
                }
                for (int64_t batch_idx = start; batch_idx < end; ++batch_idx) {
                    int64_t dim0_start = batch_idx * k_batch_size;
                    int64_t dim0_limit = batch_idx * k_batch_size + k_batch_size;
                    uint64_t infer_post_cookie = infer_post_cookie_vec[batch_idx];
                    status = infer_wait(infer_post_cookie);
                    if (!status.ok()) {
                        INFERENTIA_OP_ERROR(context, status);
                    }
                    for (auto idx = 0; idx < context->num_outputs(); ++idx) {
                        if (is_batch_output_tensors[idx]) {
                            StringPiece kaena_data = output_tensors_[idx].tensor_data();
                            Tensor slice = batch_output_tensors[idx]->Slice(
                                dim0_start, std::min(dim0_limit, batch_size));
                            status = tensor_memcpy(&slice, kaena_data,
                                                   slice.tensor_data().size());
                            if (!status.ok()) {
                                INFERENTIA_OP_ERROR(context, status);
                            }
                        }
                    }
                }

                // shift window
                start += infer_queue_length_;
                end = std::min(start + infer_queue_length_, num_batches);
            }
            timestamps.mark_below_krtd_infer();
        }
    } else {
        profile_start_session();
        status = infer(input_tensors, &timestamps);
        profile_stop_session();
        if (status.ok()) {
            for (auto idx = 0; idx < context->num_outputs(); ++idx) {
                context->set_output(idx, output_tensors_[idx]);
            }
        }
    }
    if (!status.ok()) {
        INFERENTIA_OP_ERROR(context, status);
    }
    timestamps.mark_exit();
    KAENALOG(1) << timestamps.timing_string();
}


tensorflow::Status InferentiaOp::start_model() {
    std::lock_guard<std::mutex> guard(*tpb_group_->get_mutex_infer());

    grpc::Status status;
    if (!tpb_group_->nn_is_running(krt_nn_id_) && tpb_group_->some_nn_is_running()) {
        // if krt_nn_id_ is not running, stop the current running model
        grpc::ClientContext context;
        krt::stop_request stop_request;
        stop_request.mutable_h_nn()->set_id(tpb_group_->nn_get_current_running());
        krt::stop_response stop_response;
        status = stub_->stop(&context, stop_request, &stop_response);
        KRTD_CHECK_RETURN("stop", status, stop_response);
        tpb_group_->nn_set_current_running(KRT_INVALID_NN_ID);
    }
    if (!tpb_group_->some_nn_is_running()) {
        // if no model is running, start krt_nn_id_
        grpc::ClientContext context;
        krt::start_request start_request;
        start_request.mutable_h_nn()->set_id(krt_nn_id_);
        krt::start_response start_response;
        status = stub_->start(&context, start_request, &start_response);
        KRTD_CHECK_RETURN("start", status, start_response);
        tpb_group_->nn_set_current_running(krt_nn_id_);
    }
    return tensorflow::Status::OK();
}


template<typename... Args>
tensorflow::Status subprocess_run(Args... args) {
    pid_t fork_pid;
    if ((fork_pid = fork()) < 0) {
        KAENA_SYS_ERROR_STATUS("fork", errno);
    }
    if (0 == fork_pid) {
        execlp(args..., (char*)NULL);
        KAENA_ERROR("execlp failed");
        _exit(1);
    } else {
        int status;
        if (waitpid(fork_pid, &status, 0) < 0) {
            KAENA_SYS_ERROR_STATUS("waitpid", errno);
        }
        if (!(WIFEXITED(status) && 0 == WEXITSTATUS(status))) {
            KAENA_ERROR_STATUS("child process did not exit gracefully");
        }
    }
    return tensorflow::Status::OK();
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


void InferentiaOp::profile_dump_info(const std::string &graph_def, const std::string &executable) {
    std::string filename_base = profile_dir_ + "/" + mangle_op_name(op_name_);
    std::string filename_pb = filename_base + ".pb";
    std::string filename_kelp = filename_base + ".kelp";
    std::ofstream(filename_pb, std::ios::binary) << graph_def;
    std::ofstream(filename_kelp, std::ios::binary) << executable;
}


tensorflow::Status InferentiaOp::profile_start_session() {
    if (profile_enabled_) {
        std::string new_op_name = mangle_op_name(op_name_);
        std::ostringstream filename_stream;
        filename_stream << profile_dir_ << "/" << new_op_name << "-"
                        << krt_nn_id_ << "-" << profile_session_id_ << ".ipd";
        profile_session_filename_ = filename_stream.str();
        std::ostringstream cmd_stream;
        cmd_stream << "neuron-profile start-session -s " << profile_session_filename_
                   << " -k " << krtd_server_ << " " << krt_nn_id_;
        KAENALOG(1) << "Starting profiling session by " << cmd_stream.str();
        std::ostringstream krt_nn_id_stream;
        krt_nn_id_stream << krt_nn_id_;
        tensorflow::Status status = subprocess_run(
            "neuron-profile", "neuron-profile", "start-session", "-s",
            profile_session_filename_.c_str(), "-k", krtd_server_.c_str(),
            krt_nn_id_stream.str().c_str());
        if (!status.ok()) {
            profile_session_filename_ = "";
            LOG(WARNING) << "neuron-profile start-session failed. "
                         << "Did you install aws-neuron-tools-core?";
            return status;
        }
        profile_session_id_++;
    }
    return tensorflow::Status::OK();
}


void InferentiaOp::profile_stop_session() {
    if (profile_enabled_ && "" != profile_session_filename_) {
        std::ostringstream cmd_stream;
        cmd_stream << "neuron-profile stop-session -s " << profile_session_filename_;
        KAENALOG(1) << "Stopping profiling session by " << cmd_stream.str();
        tensorflow::Status status = subprocess_run(
            "neuron-profile", "neuron-profile", "stop-session", "-s",
            profile_session_filename_.c_str());
        if (!status.ok()) {
            KAENA_ERROR("neuron-profile stop-session failed");
        }
        profile_session_filename_ = "";
    }
}


tensorflow::Status InferentiaOp::infer(
        const std::vector<const Tensor*> &input_tensors,
        FALTimestamps *timestamps) {
    if (!ready_) {
        KAENA_ERROR_STATUS("not ready for inference");
    }

    std::lock_guard<std::mutex> guard(*tpb_group_->get_mutex_infer());

    // set input tensors
    if (input_tensors.size() != input_names_.size()) {
        KAENA_ERROR_STATUS("vector sizes mismatch, input_tensors size ",
                           input_tensors.size(), ", input_names_ size",
                           input_names_.size());
    }
    krt::infer_request infer_request;
    for (size_t idx = 0; idx < input_names_.size(); ++idx) {
        krt::infer_io *infer_io = infer_request.add_ifmap();
        infer_io->set_name(input_names_[idx]);
        StringPiece tensor_data(input_tensors[idx]->tensor_data());
        if (tensor_data.size() != input_tensor_sizes_[idx]) {
            KAENA_ERROR_STATUS("incorrect input tensor size ", tensor_data.size(),
                               " with ", input_names_[idx],
                               " (", input_tensor_sizes_[idx], ")");
        }
        void *data = (void*)tensor_data.data();
        if (use_shared_memory_) {
            SharedMemory* shm;
            std::unordered_map<void*, SharedMemory*> *ptr2shm = tpb_group_->get_ptr2shm();
            if (ptr2shm->find(data) != ptr2shm->end()) {
                shm = ptr2shm->at(data);
            } else {
                std::memcpy(input_shms_[idx].ptr(), data, tensor_data.size());
                shm = &input_shms_[idx];
            }
            infer_io->mutable_buf_shm()->set_path(shm->name());
        } else {
            infer_io->set_buf(data, tensor_data.size());
        }
    }
    if (use_shared_memory_) {
        for (size_t idx = 0; idx < output_names_.size(); ++idx) {
            krt::infer_io *infer_io = infer_request.add_shm_ofmap();
            infer_io->set_name(output_names_[idx]);
            infer_io->mutable_buf_shm()->set_path(output_shms_[idx].name());
        }
    }
    infer_request.mutable_h_nn()->set_id(krt_nn_id_);

    // infer
    grpc::ClientContext context;
    krt::infer_response infer_response;
    timestamps->mark_above_krtd_infer();
    grpc::Status status = stub_->infer(&context, infer_request, &infer_response);
    timestamps->mark_below_krtd_infer();
    if (status.ok()) {
        int code = infer_response.status().code();
        // ignore inf/nan errors
        if (krt::kerr::KERR_INFER_COMPLETED_WITH_NUM_ERR == code) {
            infer_response.mutable_status()->set_code(krt::kerr::KERR_OK);
        }
    }
    KRTD_CHECK_RETURN("infer", status, infer_response);

    // output tensors are already in place if using shared memory
    if (use_shared_memory_) {
        return tensorflow::Status::OK();
    }

    // set output tensors
    std::vector<StringPiece> raw_output_tensors;
    std::unordered_map<std::string, StringPiece> map_name_raw;
    for (const auto &infer_io : infer_response.ofmap()) {
        map_name_raw.emplace(infer_io.name(), infer_io.buf());
    }
    for (const auto &out_name : output_names_) {
        if (map_name_raw.find(out_name) == map_name_raw.end()) {
            KAENA_ERROR_STATUS(
                "tensor name", out_name, " not found in infer_response.ofmap()");
        }
        raw_output_tensors.push_back(map_name_raw[out_name]);
    }
    for (size_t idx = 0; idx < output_names_.size(); ++idx) {
        StringPiece out_tensor_raw = raw_output_tensors[idx];
        Tensor *out_tensor = &output_tensors_[idx];
        tensorflow::Status tf_status = tensor_memcpy(out_tensor, out_tensor_raw);
        if (!tf_status.ok()) {
            KAENA_ERROR_STATUS("tensor_memcpy failure on tensor name: ",
                               output_names_[idx], " with error message ",
                               tf_status.error_message());
        }
    }

    return tensorflow::Status::OK();
}


tensorflow::Status InferentiaOp::infer_post(
        uint64_t *infer_post_cookie, const std::vector<const Tensor*> &input_tensors) {
    if (!ready_) {
        KAENA_ERROR_STATUS("not ready for inference");
    }

    // set input tensors
    if (input_tensors.size() != input_names_.size()) {
        KAENA_ERROR_STATUS("vector sizes mismatch, input_tensors size ",
                           input_tensors.size(), ", input_names_ size",
                           input_names_.size());
    }

    krt::infer_request infer_request;
    for (size_t idx = 0; idx < input_names_.size(); ++idx) {
        krt::infer_io *infer_io = infer_request.add_ifmap();
        infer_io->set_name(input_names_[idx]);
        StringPiece tensor_data(input_tensors[idx]->tensor_data());
        if (tensor_data.size() != input_tensor_sizes_[idx]) {
            KAENA_ERROR_STATUS("incorrect input tensor size ", tensor_data.size(),
                               " with ", input_names_[idx],
                               " (", input_tensor_sizes_[idx], ")");
        }
        void *data = (void*)tensor_data.data();
        infer_io->set_buf(data, tensor_data.size());
    }
    infer_request.mutable_h_nn()->set_id(krt_nn_id_);

    // infer
    grpc::ClientContext context;
    krt::infer_post_response infer_post_response;
    grpc::Status status = stub_->infer_post(&context, infer_request, &infer_post_response);
    KRTD_CHECK_RETURN("infer_post", status, infer_post_response);
    *infer_post_cookie = infer_post_response.cookie();
    return tensorflow::Status::OK();
}


tensorflow::Status InferentiaOp::infer_wait(uint64_t infer_post_cookie) {
    if (!ready_) {
        KAENA_ERROR_STATUS("not ready for inference");
    }

    krt::infer_wait_request infer_wait_request;
    infer_wait_request.set_cookie(infer_post_cookie);

    // infer_wait
    grpc::ClientContext context;
    krt::infer_response infer_response;
    grpc::Status status = stub_->infer_wait(&context, infer_wait_request, &infer_response);
    KRTD_CHECK_RETURN("infer_wait", status, infer_response);

    // set output tensors
    std::vector<StringPiece> raw_output_tensors;
    std::unordered_map<std::string, StringPiece> map_name_raw;
    for (const auto &infer_io : infer_response.ofmap()) {
        map_name_raw.emplace(infer_io.name(), infer_io.buf());
    }
    for (const auto &out_name : output_names_) {
        if (map_name_raw.find(out_name) == map_name_raw.end()) {
            KAENA_ERROR_STATUS(
                "tensor name", out_name, " not found in infer_response.ofmap()");
        }
        raw_output_tensors.push_back(map_name_raw[out_name]);
    }
    for (size_t idx = 0; idx < output_names_.size(); ++idx) {
        StringPiece out_tensor_raw = raw_output_tensors[idx];
        Tensor *out_tensor = &output_tensors_[idx];
        tensorflow::Status tf_status = tensor_memcpy(out_tensor, out_tensor_raw);
        if (!tf_status.ok()) {
            KAENA_ERROR_STATUS("tensor_memcpy failure on tensor name: ",
                               output_names_[idx], " with error message ",
                               tf_status.error_message());
        }
    }
    return tensorflow::Status::OK();
}


REGISTER_KERNEL_BUILDER(Name("InferentiaOp").Device(DEVICE_CPU), InferentiaOp);

}  // namespace kaena
}  // namespace tensorflow

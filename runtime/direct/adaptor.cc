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

#include "adaptor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "../macros.h"
#include "../version.h"
#include "nrt/nrt.h"
#include "nrt/nrt_experimental.h"
#include "nrt/nrt_profile.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace neuron {

#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
static std::string NrtErrMsg(NRT_STATUS rt_status) {
  static std::unordered_map<NRT_STATUS, std::string> status_to_msg({
      {NRT_SUCCESS, ""},
      {NRT_FAILURE, "Non specific failure"},
      {NRT_INVALID, "Invalid argument"},
      {NRT_INVALID_HANDLE, "Invalid handle"},
      {NRT_RESOURCE, "Failed to allocate a resource for requested operation"},
      {NRT_TIMEOUT, "Operation timed out"},
      {NRT_HW_ERROR, "Hardware failure"},
      // skipping NRT_QUEUE_FULL
      {NRT_LOAD_NOT_ENOUGH_NC,
       "Failed to allocate enough NeuronCores for loading a NEFF"},
      {NRT_UNSUPPORTED_NEFF_VERSION, "Unsupported version of NEFF"},
      {NRT_FAIL_HOST_MEM_ALLOC, "Not enough host memory"},
      {NRT_EXEC_BAD_INPUT, "Invalid input has been submitted to exec()"},
      {NRT_EXEC_COMPLETED_WITH_NUM_ERR,
       "Execution was completed with numerical errors (produced NaN)"},
      {NRT_EXEC_COMPLETED_WITH_ERR,
       "Execution was completed with other errors"},
  });
  std::string msg;
  if (status_to_msg.count(rt_status)) {
    msg = status_to_msg.at(rt_status);
  }
  Status err =
      errors::Internal(" status=", rt_status, ", error message=\"", msg, "\".");
  return err.error_message();
}

#define NRT_RETURN_IF_COND(cond, rt_status, error_fn, ...)            \
  {                                                                   \
    if (TF_PREDICT_FALSE(cond)) {                                     \
      return error_fn(__FILE__, ":", __LINE__, ":", __VA_ARGS__, ":", \
                      NrtErrMsg(rt_status));                          \
    }                                                                 \
  }
#define NRT_RETURN_IF_ERROR(rt_status, ...) \
  NRT_RETURN_IF_COND((rt_status != NRT_SUCCESS), rt_status, __VA_ARGS__)
#define NRT_RETURN_IF(rt_status, error_rt_status, ...) \
  NRT_RETURN_IF_COND((error_rt_status == rt_status), rt_status, __VA_ARGS__)
#define NRT_RETURN_IF_INVALID(obj)                                           \
  if (TF_PREDICT_FALSE(nullptr == (obj).raw_)) {                             \
    return errors::InvalidArgument(__func__, " called on invalid ", (#obj)); \
  }
#define NRT_RETURN_IF_INVALID_PTR(ptr) \
  TFN_RETURN_IF_NULLPTR(ptr);          \
  NRT_RETURN_IF_INVALID(*(ptr));
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE

Status Nrt::Init() {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  NRT_STATUS rt_status = nrt_init(/*framework=*/NRT_FRAMEWORK_TYPE_TENSORFLOW,
                                  /*fw_version=*/TF_VERSION_STRING,
                                  /*fal_version=*/TFN_VERSION_STRING);
  // To support cross-compilation, because there is not yet a reserved
  // error code for "runtime unavailable", for now we have to return
  // any error as errors::Unavailable,
  NRT_RETURN_IF_ERROR(rt_status, errors::Unavailable,
                      "Nrt::Init failed: nrt_init");
  VLOG(1) << "Nrt::Init OK";
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::GetCoreCount(int32_t *nc_count) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  TFN_RETURN_IF_NULLPTR(nc_count);
  uint32_t rt_nc_count = 0;
  NRT_STATUS rt_status = nrt_get_visible_nc_count(&rt_nc_count);
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "Nrt::GetCoreCount failed: nrt_get_visible_nc_count");
  *nc_count = (int32_t)rt_nc_count;
  VLOG(1) << "Nrt::GetCoreCount OK";
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::Close() {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  nrt_close();
  VLOG(1) << "Nrt::Close OK";
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::AllocEmptyBuffer(NrtBuffer* buffer) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  TFN_RETURN_IF_NULLPTR(buffer);
  NRT_STATUS rt_status = nrt_tensor_allocate_empty(
      /*name=*/NULL, /*tensor=*/(nrt_tensor_t**)&buffer->raw_);
  if (TF_PREDICT_FALSE(rt_status != NRT_SUCCESS)) {
    buffer->raw_ = nullptr;
  }
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "AllocEmptyBuffer failed: nrt_tensor_allocate_empty");
  VLOG(1) << "Nrt::AllocEmptyBuffer OK " << buffer->raw_;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::AttachCpuToBuffer(NrtBuffer* buffer,
                              void* cpu_buffer, size_t size) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  TFN_RETURN_IF_NULLPTR(buffer);
  TFN_RETURN_IF_NULLPTR(cpu_buffer);
  TFN_RETURN_IF_ZERO_SIZE(size);
  NRT_STATUS rt_status = nrt_tensor_attach_buffer(
      /*tensor=*/(nrt_tensor_t*)buffer->raw_,
      /*buffer=*/cpu_buffer, /*size=*/size);
  if (TF_PREDICT_FALSE(rt_status != NRT_SUCCESS)) {
    buffer->raw_ = nullptr;
  }
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "AttachCpuToBuffer failed: nrt_tensor_attach_buffer");
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::AllocHostBuffer(NrtBuffer* buffer, size_t size) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  TFN_RETURN_IF_NULLPTR(buffer);
  TFN_RETURN_IF_ZERO_SIZE(size);
  constexpr int NRT_HOST_NEURON_CORE_ID = -1;  // TODO: get from nrt.h
  NRT_STATUS rt_status = nrt_tensor_allocate(
      /*tensor_placement=*/NRT_TENSOR_PLACEMENT_VIRTUAL,
      /*logical_nc_id=*/NRT_HOST_NEURON_CORE_ID, /*size=*/size, /*name=*/NULL,
      /*tensor=*/(nrt_tensor_t**)&buffer->raw_);
  // Possible outcomes:
  //  NRT_SUCCESS: ok
  //  NRT_RESOURCE: out of host memory
  //  NRT_FAIL_HOST_MEM_ALLOC: out of host memory
  //  Others: treat as internal error
  if (TF_PREDICT_FALSE(rt_status != NRT_SUCCESS)) {
    buffer->raw_ = nullptr;
  }
#define NRT_ALLOC_RESOURCE(error_rt_status)                              \
  NRT_RETURN_IF(rt_status, (error_rt_status), errors::ResourceExhausted, \
                "Not enough host memory for allocating size=", size,     \
                ": nrt_tensor_allocate")
  NRT_ALLOC_RESOURCE(NRT_RESOURCE);
  NRT_ALLOC_RESOURCE(NRT_FAIL_HOST_MEM_ALLOC);
#undef NRT_ALLOC_RESOURCE
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "AllocHostBuffer failed: nrt_tensor_allocate");
  VLOG(1) << "Nrt::AllocHostBuffer OK " << buffer->raw_;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::FreeBuffer(NrtBuffer* buffer) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  NRT_RETURN_IF_INVALID_PTR(buffer);
  VLOG(1) << "Nrt::FreeBuffer " << buffer->raw_;
  nrt_tensor_free((nrt_tensor_t**)&buffer->raw_);
  buffer->raw_ = nullptr;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::CopyCpuToBuffer(NrtBuffer* buffer, size_t offset,
                            const void* cpu_buffer, size_t size) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  NRT_RETURN_IF_INVALID_PTR(buffer);
  TFN_RETURN_IF_NULLPTR(cpu_buffer);
  TFN_RETURN_IF_ZERO_SIZE(size);
  NRT_STATUS rt_status = nrt_tensor_write(
      /*tensor=*/(nrt_tensor_t*)buffer->raw_,
      /*buf=*/cpu_buffer, offset, size);
  // Possible outcomes:
  //  NRT_SUCCESS: ok
  //  NRT_INVALID: invalid offset
  //  Others: treat as internal error
  NRT_RETURN_IF(rt_status, NRT_INVALID, errors::InvalidArgument,
                "Invalid offset=", offset, ": nrt_tensor_write");
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "CopyCpuToBuffer failed: nrt_tensor_write");
  VLOG(1) << "Nrt::CopyCpuToBuffer OK " << cpu_buffer << " -> " << buffer->raw_;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::CopyBufferToCpu(void* cpu_buffer, size_t size,
                            const NrtBuffer& buffer, size_t offset) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  TFN_RETURN_IF_NULLPTR(cpu_buffer);
  TFN_RETURN_IF_ZERO_SIZE(size);
  NRT_RETURN_IF_INVALID(buffer);
  NRT_STATUS rt_status = nrt_tensor_read(
      /*tensor=*/(const nrt_tensor_t*)buffer.raw_,
      /*buf=*/cpu_buffer, offset, size);
  // Possible outcomes:
  //  NRT_SUCCESS: ok
  //  NRT_INVALID: invalid offset
  //  Others: treat as internal error
  NRT_RETURN_IF(rt_status, NRT_INVALID, errors::InvalidArgument,
                "Invalid offset=", offset, ": nrt_tensor_read");
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "CopyBufferToCpu failed: nrt_tensor_read");
  VLOG(1) << "Nrt::CopyBufferToCpu OK" << buffer.raw_ << " -> " << cpu_buffer;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::Load(NrtModel* model, StringPiece executable, int32_t start_nc,
                 int32_t nc_count) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  // Note: this function performs very little error checking
  // on start_nc and nc_count as it has no idea on device placement.
  TFN_RETURN_IF_NULLPTR(model);
  TFN_CHECK_ARG(start_nc < 0, "Invalid start_nc=", start_nc);
  TFN_CHECK_ARG(nc_count < 0, "Invalid nc_count=", nc_count);
  TFN_CHECK_ARG(executable.empty(), "Invalid empty executable");
  NRT_STATUS rt_status =
      nrt_load(/*neff_bytes=*/executable.data(),
               /*size=*/executable.size(), start_nc, nc_count,
               /*nrt_model_t **model=*/(nrt_model_t**)&model->raw_);
  // Possible outcomes:
  //  NRT_SUCCESS: ok
  //  NRT_INVALID: invalid NEFF
  //  NRT_INVALID_HANDLE: invalid EG (runtime internal error)
  //  NRT_RESOURCE: out of memory
  //  NRT_LOAD_NOT_ENOUGH_NC: not enough NeuronCores
  //  NRT_UNSUPPORTED_NEFF_VERSION: unsupported NEFF version
  //  NRT_FAIL_HOST_MEM_ALLOC: not enough host memory resource
  //  Others: treat as internal error
  if (TF_PREDICT_FALSE(rt_status != NRT_SUCCESS)) {
    model->raw_ = nullptr;
  }
#define NRT_LOAD_RETURN(...) NRT_RETURN_IF(rt_status, __VA_ARGS__, ": nrt_load")
  NRT_LOAD_RETURN(NRT_INVALID, errors::InvalidArgument,
                  "NEFF is invalid; it could be broken or hacked")
  NRT_LOAD_RETURN(NRT_UNSUPPORTED_NEFF_VERSION, errors::InvalidArgument);
  NRT_LOAD_RETURN(NRT_FAIL_HOST_MEM_ALLOC, errors::ResourceExhausted);
  NRT_LOAD_RETURN(NRT_LOAD_NOT_ENOUGH_NC, errors::ResourceExhausted);
  NRT_LOAD_RETURN(NRT_RESOURCE, errors::ResourceExhausted,
                  "Not enough runtime resource to load NEFF");
#undef NRT_LOAD_RETURN
  if (TF_PREDICT_FALSE(rt_status != NRT_SUCCESS)) {
    std::string nc_range;
    nc_range = std::to_string(start_nc);
    if (nc_count > 1) {
      nc_range += "-";
      nc_range += std::to_string(start_nc + nc_count - 1);
    }
    return errors::Internal("Nrt::Load failed on NeuronCores ", nc_range,
                            ": nrt_load", NrtErrMsg(rt_status));
  }
  VLOG(1) << "Nrt::Load OK " << model->raw_ << ", start_nc=" << start_nc
          << ", nc_count=" << nc_count;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::Unload(const NrtModel& model) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  NRT_RETURN_IF_INVALID(model);
  VLOG(1) << "Nrt::Unload " << model.raw_;
  NRT_STATUS rt_status = nrt_unload((nrt_model_t*)model.raw_);
  // Possible outcomes:
  //  NRT_SUCCESS: ok
  //  NRT_INVALID: invalid model pointer
  //  Others: treat as internal error
  NRT_RETURN_IF(rt_status, NRT_INVALID, errors::InvalidArgument,
                "Invalid model hancle ", model.raw_, ": nrt_unload");
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal, "Unload failed: nrt_unload");
  VLOG(1) << "Nrt::Unload OK " << model.raw_;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::AllocBufferMap(NrtBufferMap* map) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  TFN_RETURN_IF_NULLPTR(map);
  NRT_STATUS rt_status =
      nrt_allocate_tensor_set((nrt_tensor_set_t**)&map->raw_);
  // Possible outcomes:
  //  NRT_SUCCESS: ok
  //  NRT_RESOURCE: out of memory
  //  Others: treat as internal error
  if (TF_PREDICT_FALSE(rt_status != NRT_SUCCESS)) {
    map->raw_ = nullptr;
  }
  NRT_RETURN_IF(rt_status, NRT_RESOURCE, errors::ResourceExhausted,
                "Not enough runtime resource to allocate tensor set <",
                map->raw_, ">: nrt_allocate_tensor_set");
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "AllocBufferMap failed: nrt_allocate_tensor_set");
  VLOG(1) << "Nrt::AllocBufferMap OK " << map->raw_;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::FreeBufferMap(const NrtBufferMap& map) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  NRT_RETURN_IF_INVALID(map);
  VLOG(1) << "Nrt::FreeBufferMap " << map.raw_;
  nrt_destroy_tensor_set((nrt_tensor_set_t**)&map.raw_);
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::BufferMapAdd(NrtBufferMap* map, const std::string& name,
                         const NrtBuffer& buffer) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  NRT_RETURN_IF_INVALID_PTR(map);
  NRT_RETURN_IF_INVALID(buffer);
  void* raw_set = map->raw_;
  void* raw_buffer = buffer.raw_;
  NRT_STATUS rt_status = nrt_add_tensor_to_tensor_set(
      /*tensor_set=*/(nrt_tensor_set_t*)raw_set,
      /*tensor_name=*/name.c_str(),
      /*tensor=*/(nrt_tensor_t*)raw_buffer);
  // Possible outcomes:
  //  NRT_SUCCESS: ok
  //  NRT_RESOURCE: out of memory
  //  Others: treat as internal error
  NRT_RETURN_IF(rt_status, NRT_RESOURCE, errors::ResourceExhausted,
                "Not enough runtime resource to add tensor \"", name, "\" <",
                raw_buffer, ">", " to set <", raw_set,
                ">: nrt_add_tensor_to_tensor_set");
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "BufferMapAdd failed: nrt_add_tensor_to_tensor_set");
  VLOG(1) << "Nrt::BufferMapAdd OK " << map->raw_ << ", name=" << name
          << ", buffer=" << buffer.raw_;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::BufferMapGet(NrtBuffer* buffer, const NrtBufferMap& map,
                         const std::string& name) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  NRT_RETURN_IF_INVALID(map);
  NRT_RETURN_IF_INVALID_PTR(buffer);
  NRT_STATUS rt_status = nrt_get_tensor_from_tensor_set(
      /*tensor_set=*/(nrt_tensor_set_t*)map.raw_,
      /*tensor_name=*/name.c_str(),
      /*tensor=*/(nrt_tensor_t**)buffer->raw_);
  // Possible outcomes:
  //  NRT_SUCCESS: ok
  //  NRT_INVALID: ht handle invalid (runtime internal error)
  //  Others: treat as internal error
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "BufferMapGet failed: nrt_get_tensor_from_tensor_set");
  VLOG(1) << "Nrt::BufferMapGet OK " << map.raw_ << ", name=" << name
          << ", buffer=" << buffer->raw_;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::Execute(const NrtModel& model, const NrtBufferMap& input_map,
                    NrtBufferMap* output_map) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  NRT_RETURN_IF_INVALID(model);
  NRT_RETURN_IF_INVALID(input_map);
  NRT_RETURN_IF_INVALID_PTR(output_map);
  NRT_STATUS rt_status = nrt_execute(
      /*model=*/(nrt_model_t*)model.raw_,
      /*input_set=*/(const nrt_tensor_set_t*)input_map.raw_,
      /*output_set=*/(nrt_tensor_set_t*)output_map->raw_);
  // Possible outcomes:
  //  NRT_SUCCESS: ok
  //  NRT_INVALID: invalid input_map
  //  NRT_INVALID_HANDLE: invalid model handle or input handle
  //  NRT_EXEC_BAD_INPUT: invalid input
  //  NRT_EXEC_COMPLETED_WITH_NUM_ERR: possible numerical error (NaN)
  //  NRT_EXEC_COMPLETED_WITH_ERR: execution was completed with other errors
  //  Others: treat as internal error
#define NRT_EXEC_RETURN(...) \
  NRT_RETURN_IF(rt_status, __VA_ARGS__, ": nrt_execute")
  NRT_EXEC_RETURN(NRT_INVALID, errors::InvalidArgument, "Invalid input_map");
  NRT_EXEC_RETURN(NRT_INVALID_HANDLE, errors::InvalidArgument,
                  "Invalid model/input handle");
  NRT_EXEC_RETURN(NRT_EXEC_BAD_INPUT, errors::InvalidArgument);
#undef NRT_EXEC_RETURN
  if (TF_PREDICT_FALSE(NRT_EXEC_COMPLETED_WITH_NUM_ERR == rt_status)) {
    // Ignore NaN alarms
    rt_status = NRT_SUCCESS;
  }
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "Execute failed: nrt_execute");
  VLOG(1) << "Nrt::Execute OK " << model.raw_
          << ", input_map=" << input_map.raw_
          << ", output_map=" << output_map->raw_;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::ProfileStart(const NrtModel& model, const char* filename) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  NRT_RETURN_IF_INVALID(model);
  VLOG(1) << "Nrt::ProfileStart " << model.raw_;
  NRT_STATUS rt_status = nrt_profile_start((nrt_model_t*)model.raw_, filename);
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "ProfileStart failed: nrt_profile_start");
  VLOG(1) << "Nrt::ProfileStart OK " << model.raw_ << ", filename=" << filename;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

Status Nrt::ProfileStop(const char* filename) {
#ifndef AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
  VLOG(1) << "Nrt::ProfileStop";
  NRT_STATUS rt_status = nrt_profile_stop(filename);
  NRT_RETURN_IF_ERROR(rt_status, errors::Internal,
                      "ProfileStop failed: nrt_profile_stop");
  VLOG(1) << "Nrt::ProfileStart OK "
          << ", filename=" << filename;
  return Status::OK();
#else
  return errors::Unimplemented(__func__);
#endif  // AWS_NEURON_RUNTIME_LIBRARY_UNAVAILABLE
}

#undef NRT_RETURN_IF_INVALID_PTR
#undef NRT_RETURN_IF_INVALID
#undef NRT_RETURN_IF
#undef NRT_RETURN_IF_ERROR
#undef NRT_RETURN_IF_COND

}  // namespace neuron
}  // namespace tensorflow

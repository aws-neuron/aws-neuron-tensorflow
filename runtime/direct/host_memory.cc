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

#include "host_memory.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "adaptor.h"
#include "executable_info.h"
#include "macros.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace neuron {

NeuronHostBuffer::NeuronHostBuffer(size_t size) {
  // Allocate a new host buffer
  if (TF_PREDICT_FALSE(0 == size)) {
    return;
  }
  Status status = Nrt::AllocHostBuffer(&rt_buffer_, size);
  if (TF_PREDICT_FALSE(!status.ok())) {
    LOG(ERROR) << status;
    return;
  }
  size_ = size;
  payload_ = size;
}

NeuronHostBuffer::NeuronHostBuffer(void* cpu_buffer, size_t size) {
  // Donate a CPU buffer
  if (TF_PREDICT_FALSE(0 == size)) {
    return;
  }
  Status status = Nrt::AllocEmptyBuffer(&rt_buffer_);
  if (TF_PREDICT_FALSE(!status.ok())) {
    LOG(ERROR) << status;
    return;
  }
  size_ = size;
  status = Nrt::AttachCpuToBuffer(&rt_buffer_, cpu_buffer, size);
  if (TF_PREDICT_FALSE(!status.ok())) {
    LOG(ERROR) << status;
    return;
  }
}

NeuronHostBuffer::~NeuronHostBuffer() {
  if (TF_PREDICT_FALSE(0 == size_)) {
    return;
  }
  Status status = Nrt::FreeBuffer(&rt_buffer_);
  if (TF_PREDICT_FALSE(!status.ok())) {
    LOG(ERROR) << status;
    return;
  }
}

Status NeuronHostBuffer::GetStatus() {
  TFN_RETURN_IF_ZERO_SIZE(size_);
  return Status::OK();
}

Status NeuronHostBuffer::CopyCpuToBuffer(const void* cpu_buffer, size_t size,
                                         size_t offset) {
  TFN_RETURN_IF_NULLPTR(cpu_buffer);
  TFN_RETURN_IF_ZERO_SIZE(size);
  TFN_RETURN_IF_ZERO_SIZE(size_);
  return Nrt::CopyCpuToBuffer(&rt_buffer_, offset, cpu_buffer, size);
}

Status NeuronHostBuffer::CopyBufferToCpu(void* cpu_buffer, size_t size,
                                         size_t offset) {
  TFN_RETURN_IF_NULLPTR(cpu_buffer);
  TFN_RETURN_IF_ZERO_SIZE(size);
  TFN_RETURN_IF_ZERO_SIZE(size_);
  return Nrt::CopyBufferToCpu(cpu_buffer, size, rt_buffer_, offset);
}

NeuronHostBufferMap::NeuronHostBufferMap() {
  status_ = Nrt::AllocBufferMap(&rt_buffer_map_);
}

NeuronHostBufferMap::~NeuronHostBufferMap() {
  if (TF_PREDICT_TRUE(status_.ok())) {
    Nrt::FreeBufferMap(rt_buffer_map_);
  }
}

Status NeuronHostBufferMap::AddBuffer(const std::string& name,
                                      const NeuronHostBuffer& buffer) {
  TFN_RETURN_FAILED_PRECONDITION_IF_ERROR(status_);
  TFN_RETURN_IF_ZERO_SIZE(buffer.size_);
  return Nrt::BufferMapAdd(&rt_buffer_map_, name, buffer.rt_buffer_);
}

static size_t TensorSize(DataType dype, const TensorShapeProto& shape_proto) {
  int dtype_size = DataTypeSize(dype);
  int num_elements = TensorShape(shape_proto).num_elements();
  if (TF_PREDICT_FALSE(0 == dtype_size || num_elements < 0)) {
    // Return 0 on failure
    return 0;
  }
  return (size_t)(dtype_size * num_elements);
}

static inline void* GetData(const Tensor& tensor) {
#if TF_VERSION_LESS_THAN(2, 2)
  return (void*)(const_cast<char*>(tensor.tensor_data().data()));
#else
  return tensor.data();
#endif
}

Status NeuronHostMemory::SetupBuffers(const NeuronExecutableInfo& info,
                                      std::vector<Tensor>* input_tensors,
                                      std::vector<Tensor>* output_tensors) {
  VLOG(1) << "entering NeuronHostMemory::SetupBuffers";
  input_buffers_.reserve(info.input_dtypes.type_size());
  for (int idx = 0; idx < info.input_dtypes.type_size(); ++idx) {
    size_t size =
        TensorSize(info.input_dtypes.type(idx), info.input_shapes.shape(idx));
    std::shared_ptr<NeuronHostBuffer> buffer;
    const Tensor& tensor = input_tensors->at(idx);
    if (TF_PREDICT_TRUE(tensor.tensor_data().size() == size)) {
      buffer = std::make_shared<NeuronHostBuffer>(GetData(tensor), size);
    } else {
      buffer = std::make_shared<NeuronHostBuffer>(size);
    }
    input_buffers_.push_back(buffer);
    TF_RETURN_IF_ERROR(input_buffers_.back()->GetStatus());
  }
  VLOG(1) << "NeuronHostMemory::SetupBuffers input_buffers_ done";
  for (int idx = 0; idx < info.input_dtypes.type_size(); ++idx) {
    TF_RETURN_IF_ERROR(input_buffer_map_.AddBuffer(info.input_names.s(idx),
                                                   *input_buffers_.at(idx)));
  }
  VLOG(1) << "NeuronHostMemory::SetupBuffers input_buffer_map_ done";
  output_buffers_.reserve(info.output_dtypes.type_size());
  for (int idx = 0; idx < info.output_dtypes.type_size(); ++idx) {
    size_t size =
        TensorSize(info.output_dtypes.type(idx), info.output_shapes.shape(idx));
    std::shared_ptr<NeuronHostBuffer> buffer;
    const Tensor& tensor = output_tensors->at(idx);
    if (TF_PREDICT_TRUE(tensor.tensor_data().size() == size)) {
      buffer = std::make_shared<NeuronHostBuffer>(GetData(tensor), size);
    } else {
      buffer = std::make_shared<NeuronHostBuffer>(size);
    }
    output_buffers_.push_back(buffer);
    TF_RETURN_IF_ERROR(output_buffers_.back()->GetStatus());
  }
  VLOG(1) << "NeuronHostMemory::SetupBuffers output_buffers_ done";
  for (int idx = 0; idx < info.output_dtypes.type_size(); ++idx) {
    TF_RETURN_IF_ERROR(output_buffer_map_.AddBuffer(info.output_names.s(idx),
                                                    *output_buffers_.at(idx)));
  }
  VLOG(1) << "NeuronHostMemory::SetupBuffers done";
  return Status::OK();
}

Status NeuronHostMemory::SetupBuffersCacheWeights(
    const NeuronExecutableInfo& info, std::vector<Tensor>* input_tensors,
    std::vector<Tensor>* output_tensors) {
  VLOG(1) << "entering NeuronHostMemory::SetupBuffersCacheWeights";
  auto locs = info.real_input_locations;
  input_buffers_.reserve(locs->i_size());
  for (int idx = 0; idx < locs->i_size(); ++idx) {
    size_t size = TensorSize(info.input_dtypes.type(locs->i(idx)),
                             info.input_shapes.shape(locs->i(idx)));
    std::shared_ptr<NeuronHostBuffer> buffer;
    const Tensor& tensor = input_tensors->at(idx);
    if (TF_PREDICT_TRUE(tensor.tensor_data().size() == size)) {
      buffer = std::make_shared<NeuronHostBuffer>(GetData(tensor), size);
    } else {
      buffer = std::make_shared<NeuronHostBuffer>(size);
    }
    input_buffers_.push_back(buffer);
    TF_RETURN_IF_ERROR(input_buffers_.back()->GetStatus());
  }
  VLOG(1) << "NeuronHostMemory::SetupBuffers input_buffers_ done";
  for (int idx = 0; idx < locs->i_size(); ++idx) {
    TF_RETURN_IF_ERROR(input_buffer_map_.AddBuffer(info.input_names.s(idx),
                                                   *input_buffers_.at(idx)));
  }
  VLOG(1) << "NeuronHostMemory::SetupBuffers input_buffer_map_ done";
  output_buffers_.reserve(info.output_dtypes.type_size());
  for (int idx = 0; idx < info.output_dtypes.type_size(); ++idx) {
    size_t size =
        TensorSize(info.output_dtypes.type(idx), info.output_shapes.shape(idx));
    std::shared_ptr<NeuronHostBuffer> buffer;
    const Tensor& tensor = output_tensors->at(idx);
    if (TF_PREDICT_TRUE(tensor.tensor_data().size() == size)) {
      buffer = std::make_shared<NeuronHostBuffer>(GetData(tensor), size);
    } else {
      buffer = std::make_shared<NeuronHostBuffer>(size);
    }
    output_buffers_.push_back(buffer);
    TF_RETURN_IF_ERROR(output_buffers_.back()->GetStatus());
  }
  VLOG(1) << "NeuronHostMemory::SetupBuffers output_buffers_ done";
  for (int idx = 0; idx < info.output_dtypes.type_size(); ++idx) {
    TF_RETURN_IF_ERROR(output_buffer_map_.AddBuffer(info.output_names.s(idx),
                                                    *output_buffers_.at(idx)));
  }
  VLOG(1) << "NeuronHostMemory::SetupBuffers done";
  return Status::OK();
}

Status NeuronHostMemory::CopyCPUToInputBuffers(
    const std::vector<Tensor>& input_tensors) {
  VLOG(1) << "Starting CopyCPUToInputBuffers";
  if (TF_PREDICT_FALSE(input_tensors.size() != input_buffers_.size())) {
    return errors::InvalidArgument("Incorrect number of input tensors: given ",
                                   input_tensors.size(), ", expect ",
                                   input_buffers_.size());
  }
  for (size_t idx = 0; idx < input_buffers_.size(); ++idx) {
    const Tensor& tensor = input_tensors.at(idx);
    size_t tensor_size = tensor.tensor_data().size();
    std::shared_ptr<NeuronHostBuffer> buffer = input_buffers_.at(idx);
    if (TF_PREDICT_FALSE(buffer->GetSize() < tensor_size)) {
      return errors::InvalidArgument("Invalid input tensor size: given ",
                                     tensor.DeviceSafeDebugString(),
                                     ", expected size ", buffer->GetSize());
    }
  }
  for (size_t idx = 0; idx < input_buffers_.size(); ++idx) {
    std::shared_ptr<NeuronHostBuffer> buffer = input_buffers_.at(idx);
    if (TF_PREDICT_TRUE(!buffer->Owned())) {
      // Skip copy if buffer is donated
      continue;
    }
    const Tensor& tensor = input_tensors.at(idx);
    size_t tensor_size = tensor.tensor_data().size();
    TF_RETURN_IF_ERROR(buffer->CopyCpuToBuffer(GetData(tensor), tensor_size));
    if (TF_PREDICT_FALSE(tensor_size < buffer->GetSize())) {
      VLOG(1) << "Filling the uninitialized part of input " << idx << " with 0";
      size_t offset = tensor_size;
      size_t size = buffer->GetSize() - tensor_size;
      std::vector<uint8_t> zeros(size, 0);
      TF_RETURN_IF_ERROR(buffer->CopyCpuToBuffer(zeros.data(), size, offset));
    }
  }
  VLOG(1) << "NeuronHostMemory::CopyCPUToInputBuffers done";
  return Status::OK();
}

Status NeuronHostMemory::CopyOutputBuffersToCPU(
    const std::vector<Tensor>& output_tensors) {
  if (TF_PREDICT_FALSE(output_tensors.size() != output_buffers_.size())) {
    return errors::InvalidArgument("Incorrect number of output tensors: given ",
                                   output_tensors.size(), ", expect ",
                                   output_buffers_.size());
  }
  for (size_t idx = 0; idx < output_buffers_.size(); ++idx) {
    const Tensor& tensor = output_tensors.at(idx);
    size_t tensor_size = tensor.tensor_data().size();
    std::shared_ptr<NeuronHostBuffer> buffer = output_buffers_.at(idx);
    if (TF_PREDICT_FALSE(buffer->GetSize() < tensor_size)) {
      return errors::InvalidArgument("Invalid output tensor size: given ",
                                     tensor.DeviceSafeDebugString(),
                                     ", expected size ", buffer->GetSize());
    }
  }
  for (size_t idx = 0; idx < output_buffers_.size(); ++idx) {
    std::shared_ptr<NeuronHostBuffer> buffer = output_buffers_.at(idx);
    if (TF_PREDICT_TRUE(!buffer->Owned())) {
      VLOG(1) << "Buffer was donated";
      // Skip copy if buffer is donated
      continue;
    }
    VLOG(1) << "Calling CopyBufferToCpu with small performance impact, likely "
            << "due to shape mismatch between CPU tensor and NeuronHostBuffer";
    const Tensor& tensor = output_tensors.at(idx);
    size_t tensor_size = tensor.tensor_data().size();
    TF_RETURN_IF_ERROR(buffer->CopyBufferToCpu(GetData(tensor), tensor_size));
  }
  VLOG(1) << "NeuronHostMemory::CopyOutputBuffersToCPU done";
  return Status::OK();
}

// NEURON DEVICE BUFFER

NeuronDeviceBuffer::NeuronDeviceBuffer(size_t size, int32_t memory_id) {
  // Allocate a new device buffer
  if (TF_PREDICT_FALSE(0 == size)) {
    return;
  }
  Status status = Nrt::AllocDeviceBuffer(&rt_buffer_, size, memory_id);
  if (TF_PREDICT_FALSE(!status.ok())) {
    VLOG(1) << "There was in error in NDB constructor called with size " << size
            << " and memory_id " << memory_id;
    LOG(ERROR) << status;
    return;
  }
  size_ = size;
  payload_ = size;
  memory_id_ = memory_id;
}

NeuronDeviceBuffer::~NeuronDeviceBuffer() {
  if (TF_PREDICT_FALSE(0 == size_)) {
    return;
  }
  Status status = Nrt::FreeBuffer(&rt_buffer_);
  if (TF_PREDICT_FALSE(!status.ok())) {
    LOG(ERROR) << status;
    return;
  }
}

Status NeuronDeviceBuffer::GetStatus() {
  TFN_RETURN_IF_ZERO_SIZE(size_);
  return Status::OK();
}

Status NeuronDeviceBuffer::CopyCpuToBuffer(const void* cpu_buffer, size_t size,
                                           size_t offset) {
  TFN_RETURN_IF_NULLPTR(cpu_buffer);
  TFN_RETURN_IF_ZERO_SIZE(size);
  TFN_RETURN_IF_ZERO_SIZE(size_);
  return Nrt::CopyCpuToBuffer(&rt_buffer_, offset, cpu_buffer, size);
}

Status NeuronDeviceBuffer::CopyBufferToCpu(void* cpu_buffer, size_t size,
                                           size_t offset) {
  TFN_RETURN_IF_NULLPTR(cpu_buffer);
  TFN_RETURN_IF_ZERO_SIZE(size);
  TFN_RETURN_IF_ZERO_SIZE(size_);
  return Nrt::CopyBufferToCpu(cpu_buffer, size, rt_buffer_, offset);
}

// NEURON DEVICE BUFFER MAP

NeuronDeviceBufferMap::NeuronDeviceBufferMap() {
  status_ = Nrt::AllocBufferMap(&rt_buffer_map_);
}

NeuronDeviceBufferMap::~NeuronDeviceBufferMap() {
  if (TF_PREDICT_TRUE(status_.ok())) {
    Nrt::FreeBufferMap(rt_buffer_map_);
  }
}

Status NeuronDeviceBufferMap::AddBuffer(const std::string& name,
                                        const NeuronDeviceBuffer& buffer) {
  TFN_RETURN_FAILED_PRECONDITION_IF_ERROR(status_);
  TFN_RETURN_IF_ZERO_SIZE(buffer.size_);
  return Nrt::BufferMapAdd(&rt_buffer_map_, name, buffer.rt_buffer_);
}

// NEURON DEVICE MEMORY

Status NeuronDeviceMemory::SetupBuffers(
    const NeuronExecutableInfo& info, std::vector<Tensor>* input_tensors,
    std::vector<Tensor>* output_tensors,
    std::vector<std::shared_ptr<NeuronDeviceBuffer>>& cache,
    std::vector<int> real_input_locations) {
  VLOG(1) << "entering NeuronDeviceMemory::SetupBuffers";
  input_buffers_.reserve(info.input_dtypes.type_size());
  bool cache_setup = cache.size() > 0;
  std::shared_ptr<NeuronDeviceBuffer> buffer;
  for (int idx = 0; idx < info.input_dtypes.type_size(); ++idx) {
    size_t size =
        TensorSize(info.input_dtypes.type(idx), info.input_shapes.shape(idx));
    const Tensor& tensor = input_tensors->at(idx);
    if (!cache_setup) {
      buffer = std::make_shared<NeuronDeviceBuffer>(size, memory_id_);
      TF_RETURN_IF_ERROR(buffer->CopyCpuToBuffer(GetData(tensor), size));
      input_buffers_.push_back(buffer);
      std::shared_ptr<NeuronDeviceBuffer> cached_buffer = buffer;
      // add buffer to cache
      cache.push_back(cached_buffer);
      TF_RETURN_IF_ERROR(input_buffers_.back()->GetStatus());
    } else {
      // if the cache is setup then see if this is a real input.
      if (std::find(real_input_locations.begin(), real_input_locations.end(),
                    idx) != real_input_locations.end()) {
        VLOG(1) << "Using Real Input at index: " << idx;
        buffer = std::make_shared<NeuronDeviceBuffer>(size, memory_id_);
        TF_RETURN_IF_ERROR(buffer->CopyCpuToBuffer(GetData(tensor), size));
        input_buffers_.push_back(buffer);
      }
      // if it isn't use the cached one
      else {
        VLOG(1) << "Using Cached Buffer at index: " << idx;
        input_buffers_.push_back(cache.at(idx));
      }
      TF_RETURN_IF_ERROR(input_buffers_.back()->GetStatus());
    }
  }
  VLOG(1) << "NeuronDeviceMemory::SetupBuffers input_buffers_ done";
  for (int idx = 0; idx < info.input_dtypes.type_size(); ++idx) {
    TF_RETURN_IF_ERROR(input_buffer_map_.AddBuffer(info.input_names.s(idx),
                                                   *input_buffers_.at(idx)));
  }
  VLOG(1) << "NeuronDeviceMemory::SetupBuffers input_buffer_map_ done";
  output_buffers_.reserve(info.output_dtypes.type_size());
  for (int idx = 0; idx < info.output_dtypes.type_size(); ++idx) {
    size_t size =
        TensorSize(info.output_dtypes.type(idx), info.output_shapes.shape(idx));
    std::shared_ptr<NeuronDeviceBuffer> buffer;
    const Tensor& tensor = output_tensors->at(idx);
    buffer = std::make_shared<NeuronDeviceBuffer>(size, memory_id_);
    output_buffers_.push_back(buffer);
    TF_RETURN_IF_ERROR(output_buffers_.back()->GetStatus());
  }
  VLOG(1) << "NeuronDeviceMemory::SetupBuffers output_buffers_ done";
  for (int idx = 0; idx < info.output_dtypes.type_size(); ++idx) {
    TF_RETURN_IF_ERROR(output_buffer_map_.AddBuffer(info.output_names.s(idx),
                                                    *output_buffers_.at(idx)));
  }
  VLOG(1) << "NeuronDeviceMemory::SetupBuffers done";
  return Status::OK();
}

Status NeuronDeviceMemory::CopyOutputBuffersToCPU(
    const std::vector<Tensor>& output_tensors) {
  if (TF_PREDICT_FALSE(output_tensors.size() != output_buffers_.size())) {
    return errors::InvalidArgument("Incorrect number of output tensors: given ",
                                   output_tensors.size(), ", expect ",
                                   output_buffers_.size());
  }
  for (size_t idx = 0; idx < output_buffers_.size(); ++idx) {
    const Tensor& tensor = output_tensors.at(idx);
    size_t tensor_size = tensor.tensor_data().size();
    std::shared_ptr<NeuronDeviceBuffer> buffer = output_buffers_.at(idx);
    if (TF_PREDICT_FALSE(buffer->GetSize() < tensor_size)) {
      return errors::InvalidArgument("Invalid output tensor size: given ",
                                     tensor.DeviceSafeDebugString(),
                                     ", expected size ", buffer->GetSize());
    }
  }
  for (size_t idx = 0; idx < output_buffers_.size(); ++idx) {
    std::shared_ptr<NeuronDeviceBuffer> buffer = output_buffers_.at(idx);
    if (TF_PREDICT_TRUE(!buffer->Owned())) {
      VLOG(1) << "Buffer was donated, but I don't think I should be called";
      // Skip copy if buffer is donated
      continue;
    }
    VLOG(1) << "Calling CopyBufferToCpu with small performance impact, likely "
            << "due to shape mismatch between CPU tensor and NeuronHostBuffer";
    const Tensor& tensor = output_tensors.at(idx);
    size_t tensor_size = tensor.tensor_data().size();
    TF_RETURN_IF_ERROR(buffer->CopyBufferToCpu(GetData(tensor), tensor_size));
  }
  VLOG(1) << "NeuronDeviceMemory::CopyOutputBuffersToCPU done";
  return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow

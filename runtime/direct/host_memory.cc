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
#include "../macros.h"
#include "adaptor.h"
#include "executable_info.h"
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
  if (TF_PREDICT_FALSE(0 == size)) {
    return;
  }
  Status status = Nrt::AllocHostBuffer(&rt_buffer_, size);
  if (TF_PREDICT_FALSE(!status.ok())) {
    LOG(ERROR) << status.error_message();
    return;
  }
  size_ = size;
}

NeuronHostBuffer::~NeuronHostBuffer() {
  if (TF_PREDICT_TRUE(0 != size_)) {
    Nrt::FreeBuffer(&rt_buffer_);
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

Status NeuronHostMemory::SetupBuffers(const NeuronExecutableInfo& info) {
  VLOG(1) << "entering NeuronHostMemory::SetupBuffers";
  input_buffers_.reserve(info.input_dtypes.type_size());
  for (int idx = 0; idx < info.input_dtypes.type_size(); ++idx) {
    size_t size =
        TensorSize(info.input_dtypes.type(idx), info.input_shapes.shape(idx));
    input_buffers_.push_back(std::make_shared<NeuronHostBuffer>(size));
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
    output_buffers_.push_back(std::make_shared<NeuronHostBuffer>(size));
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

static inline void* GetData(const Tensor& tensor) {
#if TF_VERSION_LESS_THAN(2, 2)
  return (void*)(const_cast<char*>(tensor.tensor_data().data()));
#else
  return tensor.data();
#endif
}

Status NeuronHostMemory::CopyCPUToInputBuffers(
    const std::vector<Tensor>& input_tensors) {
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
    const Tensor& tensor = input_tensors.at(idx);
    size_t tensor_size = tensor.tensor_data().size();
    std::shared_ptr<NeuronHostBuffer> buffer = input_buffers_.at(idx);
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
    const Tensor& tensor = output_tensors.at(idx);
    size_t tensor_size = tensor.tensor_data().size();
    std::shared_ptr<NeuronHostBuffer> buffer = output_buffers_.at(idx);
    TF_RETURN_IF_ERROR(buffer->CopyBufferToCpu(GetData(tensor), tensor_size));
  }
  VLOG(1) << "NeuronHostMemory::CopyOutputBuffersToCPU done";
  return Status::OK();
}

}  // namespace neuron
}  // namespace tensorflow

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

#ifndef TENSORFLOW_NEURON_RUNTIME_DEVICE_H_
#define TENSORFLOW_NEURON_RUNTIME_DEVICE_H_

#include "macros.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace neuron {

const char* const DEVICE_NEURON = "AWS_NEURON";

// Neuron device implementation.
class NeuronDevice : public LocalDevice {
 public:
  NeuronDevice(const SessionOptions& options, const DeviceAttributes& attrs);
  ~NeuronDevice() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;
  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override;
#if TF_VERSION_LESS_THAN(2, 3)
  Status FillContextMap(const Graph* graph,
                        DeviceContextMap* device_context_map);
#endif
  Status TryGetDeviceContext(DeviceContext** out_context);
  Status Sync() override { return Status::OK(); }

 private:
  Allocator* cpu_allocator_;  // Not owned
};

}  // namespace neuron
}  // namespace tensorflow

#endif  // TENSORFLOW_NEURON_RUNTIME_DEVICE_H_

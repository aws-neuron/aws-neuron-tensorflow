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

#include <string>
#include <vector>

#include "device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace neuron {

class NeuronDeviceContext : public DeviceContext {
 public:
  NeuronDeviceContext() = default;

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override {
    VLOG(1) << "copying " << cpu_tensor << " to " << device_tensor
            << " in NeuronDeviceContext::CopyCPUTensorToDevice";
    *device_tensor = *cpu_tensor;
    done(Status::OK());
  }
  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             StringPiece tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override {
    VLOG(1) << "copying " << device_tensor << " to " << cpu_tensor
            << " in NeuronDeviceContext::CopyDeviceTensorToCPU";
    *cpu_tensor = *device_tensor;
    done(Status::OK());
  }
  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override {
    VLOG(1) << "copying " << input_tensor << " to " << output_tensor
            << " in NeuronDeviceContext::CopyTensorInSameDevice";
    *output_tensor = *input_tensor;
    done(Status::OK());
  }
};

NeuronDevice::NeuronDevice(const SessionOptions& options,
                           const DeviceAttributes& attrs)
    : LocalDevice(options, attrs) {
  ProcessState* ps = ProcessState::singleton();
  cpu_allocator_ = ps->GetCPUAllocator(port::kNUMANoAffinity);
}

NeuronDevice::~NeuronDevice() {}

Allocator* NeuronDevice::GetAllocator(AllocatorAttributes attr) {
  VLOG(1) << "returning cpu allocator from NeuronDevice::GetAllocator";
  return cpu_allocator_;
}

Status NeuronDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                         const AllocatorAttributes alloc_attrs,
                                         Tensor* tensor) {
  VLOG(1) << "entering NeuronDevice::MakeTensorFromProto";
  if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
    Tensor parsed(tensor_proto.dtype());
    if (parsed.FromProto(cpu_allocator_, tensor_proto)) {
      *tensor = std::move(parsed);
      return Status::OK();
    }
  }
  return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                 tensor_proto.DebugString());
}

void NeuronDevice::CopyTensorInSameDevice(const Tensor* input_tensor,
                                          Tensor* output_tensor,
                                          const DeviceContext* device_context,
                                          StatusCallback done) {
  VLOG(1) << "entering NeuronDevice::CopyTensorInSameDevice";
  if (input_tensor->NumElements() != output_tensor->NumElements()) {
    done(errors::Internal(
        "Neuron->Neuron copy shape mismatch: input=", input_tensor->shape(),
        ", output=", output_tensor->shape()));
    return;
  }
  tensor::DeepCopy(*input_tensor, output_tensor);
  done(Status::OK());
}

#if TF_VERSION_LESS_THAN(2, 3)
Status NeuronDevice::FillContextMap(const Graph* graph,
                                    DeviceContextMap* device_context_map) {
  device_context_map->resize(graph->num_node_ids());
  for (Node* n : graph->nodes()) {
    (*device_context_map)[n->id()] = new NeuronDeviceContext;
  }
  return Status::OK();
}
#endif

Status NeuronDevice::TryGetDeviceContext(DeviceContext** out_context) {
  *out_context = new NeuronDeviceContext;
  return Status::OK();
}

class NeuronDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<std::string>* devices) override {
    devices->push_back(
        strings::StrCat("/physical_device:", DEVICE_NEURON, ":0"));
    return Status::OK();
  }

  Status CreateDevices(const SessionOptions& options, const std::string& prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    std::string name = strings::StrCat(prefix, "/device:", DEVICE_NEURON, ":0");
    DeviceAttributes attr = Device::BuildDeviceAttributes(
        name, DEVICE_NEURON, Bytes(256 << 20), DeviceLocality());
    devices->push_back(absl::make_unique<NeuronDevice>(options, attr));
    VLOG(1) << "NeuronDevice " << name << " at " << devices->back().get();
    return Status::OK();
  }
};

template <class Factory>
class NeuronRegistrar {
 public:
  explicit NeuronRegistrar(const std::string& device_type) {
    int32 priority = DeviceFactory::DevicePriority(DEVICE_NEURON);
    DeviceFactory::Register(device_type, new Factory(), priority + 1);
  }
};

static NeuronRegistrar<NeuronDeviceFactory> neuron_device(DEVICE_NEURON);

}  // namespace neuron
}  // namespace tensorflow

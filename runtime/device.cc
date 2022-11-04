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
#include "engine.h"
#include "direct/env.h"
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
  NeuronEngineManager& nem = NeuronEngineManager::GetNeuronEngineManager();
  shm_allocator_ = nem.get_shm_allocator();
}

NeuronDevice::~NeuronDevice() {}

Allocator* NeuronDevice::GetAllocator(AllocatorAttributes attr) {
  Allocator* allocator;
  std::string name;
  if (TF_PREDICT_TRUE(on_shm(attr))) {
    if (TF_PREDICT_FALSE(!shm_allocator_->is_valid())) {
      LOG(ERROR) << "NeuronDevice::GetAllocator returns invalid shm allocator";
    }
    allocator = shm_allocator_;
    name = "shm";
  } else {
    allocator = cpu_allocator_;
    name = "cpu";
  }
  VLOG(1) << "NeuronDevice::GetAllocator returns " << name << " allocator";
  return allocator;
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

void NeuronDevice::set_on_shm(AllocatorAttributes* attr, bool v) {
  attr->value |= (static_cast<int>(v) << NeuronDevice::on_shm_shift_);
}

bool NeuronDevice::on_shm(const AllocatorAttributes& attr) {
  return attr.value & (0x1 << NeuronDevice::on_shm_shift_);
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
    if (nullptr == DeviceFactory::GetFactory(device_type)) {
      // only register if device_type is still unregistered
      std::string priority_str = env_get("AWS_NEURON_DEVICE_PRIORITY", "50");
      int32 priority = stoi_no_throw(priority_str);
      // priority value higher than that of GPUCompatibleCPU (70) would let
      // tensorflow runtime to dispatch ops on us automatically, which then
      // encourages XLA to try to compile for us and crash
#if TF_VERSION_LESS_THAN(2, 5)
      DeviceFactory::Register(device_type, new Factory(), priority);
#else
      DeviceFactory::Register(device_type, new Factory(), priority, false);
#endif
    }
  }
};

static NeuronRegistrar<NeuronDeviceFactory> neuron_device(DEVICE_NEURON);

}  // namespace neuron
}  // namespace tensorflow

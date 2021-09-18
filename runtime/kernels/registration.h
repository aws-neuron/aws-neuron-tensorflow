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

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace neuron {

static KernelDef* neuron_kernel(const std::string& type,
                                const std::string& device_type) {
  // Need to allocate from an existing one to workaround the bug that setting
  // one attribute can affect other attributes
  auto identity_n_kernel = GetRegisteredKernelsForOp("IdentityN").kernel(0);
  KernelDef* kernel = new KernelDef(identity_n_kernel);
  kernel->clear_op();
  kernel->clear_device_type();
  kernel->clear_constraint();
  kernel->clear_host_memory_arg();
  kernel->clear_label();
  kernel->clear_priority();
  KernelList type_kernel_list = GetRegisteredKernelsForOp(type);
  bool is_registered = false;
  for (const auto& type_kernel : type_kernel_list.kernel()) {
    if (type_kernel.device_type() == device_type) {
      is_registered = true;
      break;
    }
  }
  kernel->set_op(is_registered ? "_no_register" : type);
  kernel->set_device_type(device_type);
  return kernel;
}

#ifdef AWS_NEURON_TFSERV
#define NEURON_REGISTER_KERNEL_BUILDER(type, device_type, class_name) \
  REGISTER_KERNEL_BUILDER(Name(type).Device(device_type), class_name);
#else
#define NEURON_REGISTER_KERNEL_BUILDER(type, device_type, class_name) \
  static kernel_factory::OpKernelRegistrar                            \
      neuron_##device_type##_##class_name##_registrar(                \
          neuron_kernel((type), (device_type)), (type),               \
          [](OpKernelConstruction* context) -> OpKernel* {            \
            return new (class_name)(context);                         \
          });
#endif

}  // namespace neuron
}  // namespace tensorflow

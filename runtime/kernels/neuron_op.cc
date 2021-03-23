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

#include "neuron_op.h"
#include "../device.h"

namespace tensorflow {
namespace register_kernel {

static KernelDef* neuron_kernel(const std::string& type) {
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
  int64 kernel_size = GetRegisteredKernelsForOp(type).kernel_size();
  kernel->set_op(kernel_size ? "_no_register" : type);
  kernel->set_device_type(neuron::DEVICE_NEURON);
  return kernel;
}

class NeuronName {
 public:
  const KernelDef* Build() { return neuron_kernel("NeuronOp"); }
};

}  // namespace register_kernel

namespace neuron {

void NeuronOp::Compute(OpKernelContext* ctx) {
  std::vector<Tensor> input_tensors(ctx->num_inputs());
  for (auto idx = 0; idx < ctx->num_inputs(); ++idx) {
    input_tensors[idx] = ctx->input(idx);
  }
  OP_REQUIRES_OK(ctx, model_.compute(ctx, def(), input_tensors));
}

// need to override kernel_builder as NeuronName to prevent multiple kernel
// registrations
#if TF_VERSION_LESS_THAN(2, 4)
REGISTER_KERNEL_BUILDER(NeuronName(), neuron::NeuronOp);
#else
REGISTER_KERNEL_BUILDER_IMPL_2("NeuronOp", register_kernel::NeuronName(), false,
                               NeuronOp);
#endif

}  // namespace neuron
}  // namespace tensorflow

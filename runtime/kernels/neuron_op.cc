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
#include "registration.h"
#include "../device.h"

namespace tensorflow {
namespace neuron {

void NeuronOp::Compute(OpKernelContext* ctx) {
  std::vector<Tensor> input_tensors(ctx->num_inputs());
  for (auto idx = 0; idx < ctx->num_inputs(); ++idx) {
    input_tensors[idx] = ctx->input(idx);
  }
  OP_REQUIRES_OK(ctx, model_.compute(ctx, def(), input_tensors));
}

NEURON_REGISTER_KERNEL_BUILDER("NeuronOp", DEVICE_NEURON, NeuronOp);

}  // namespace neuron
}  // namespace tensorflow

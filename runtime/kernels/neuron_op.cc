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
#include "tensorflow/core/framework/op_kernel.h"
#include "registration.h"
#include "../device.h"
#include "../engine.h"

namespace tensorflow {
namespace neuron {

NeuronOp::NeuronOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  VLOG(1) << "NeuronOp contructor " << this;
  grpc_runtime_status_ =
      NeuronEngineManager::GetNeuronEngineManager().runtime_status();
}

void NeuronOp::Compute(OpKernelContext* ctx) {
  // Fail early if there is a precondition error
  if (grpc_runtime_status_.code() == error::Code::FAILED_PRECONDITION) {
    OP_REQUIRES_OK(ctx, grpc_runtime_status_);
  }

  // Use GRPC runtime if available and there is no precondition error
  if (grpc_runtime_status_.ok()) {
    OP_REQUIRES_OK(ctx, model_.compute(ctx, def()));
    return;
  }

  // Call direct-link runtime
  OP_REQUIRES_OK(ctx, function_.Run(ctx, def()));
}

#if TF_VERSION_LESS_THAN(2, 0)
NEURON_REGISTER_KERNEL_BUILDER("NeuronOp", DEVICE_CPU, NeuronOp);
#else
NEURON_REGISTER_KERNEL_BUILDER("NeuronOp", DEVICE_NEURON, NeuronOp);
#endif

}  // namespace neuron
}  // namespace tensorflow
